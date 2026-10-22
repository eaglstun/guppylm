# Experiment: is the fish thinking, or is it a lookup table with fins?

Branch `exp/data`. GuppyLM, 8.7M params, 10k steps at batch 32, data from
`guppylm/generate_data.py` (57,100 train / 3,000 eval). Baseline from the main checkout:
eval loss 0.385, keyword hit rate 0.775.

## Questions

1. How varied is the training data really?
2. Does the baseline model compose, or recall?
3. What happens on a smaller data diet (1K, 5K, 15K examples, same 10k steps)?
4. Anything odd in the data or harness.

## Code added

- `guppylm/dataset.py`: `subset_lines()` and a `limit` / `subset_seed` on the dataset.
  The subset is a seeded `random.sample` of train lines, so analysis can rebuild exactly
  what a run saw. Eval set is never subset.
- `guppylm/config.py`, `guppylm/train.py`: `TrainConfig.train_limit` / `subset_seed`;
  `train()` now returns `eval_history` (step, interval train loss, eval loss).
- `tools/benchmark.py`: `--train-limit N --subset-seed S`, recorded in `results.json`
  (`train_config.train_limit`, `train.epochs`, `train.eval_history`). It now also scores
  `final_model.pt` on the full eval set and both checkpoints on the train subset in eval
  mode (no dropout). `compare` shows these rows.
- `tools/data_analysis.py`: uniqueness, sentence reuse, train/eval overlap.
- `tools/memorization.py`: samples a checkpoint on CPU and classifies every reply
  against train outputs (verbatim / composed from train sentences / contains a new
  sentence), plus a topic classifier and a noun-copying probe.
- `tools/quality_many.py`: keyword hit rate with 30 samples per case (480 replies)
  instead of the harness's 5 (80 replies).

## Exact commands

```bash
cd /Users/eeaglstun/Documents/dev/guppylm-exp-data
G=/private/tmp/claude-501/-Users-eeaglstun-Documents-web-ericeaglstun-ai/f124945c-1d81-418b-a0c8-d0a95e2e49aa/scratchpad/guppy/gpu.sh

python3 tools/data_analysis.py --json runs/analysis/data_report.json

# baseline checkpoints copied from the main checkout's runs/baseline/, then re-scored
$G python3 tools/benchmark.py run --name baseline --skip-train --no-onnx
for n in 1000 5000 15000; do
  $G python3 tools/benchmark.py run --name data$((n/1000))k --train-limit $n --no-onnx
done

for spec in baseline: data1k:1000 data5k:5000 data15k:15000; do
  r=${spec%%:*}; lim=${spec#*:}
  for c in best final; do
    python3 tools/memorization.py --ckpt runs/$r/checkpoints/${c}_model.pt \
      ${lim:+--train-limit $lim} --out runs/analysis/memo_${r}_${c}.json
  done
done   # seed 1234, 3 samples per prompt, temperature 0.7, top_k 50

python3 tools/quality_many.py --samples 30 --seed 7 --out runs/analysis/quality_many.json runs/*/checkpoints/{best,final}_model.pt
```

`runs/` is gitignored; the numbers below are copied from `runs/*/results.json` and
`runs/analysis/*.json`.

---

## 1. How varied is the data? (measured)

|                                                                          | count                              |
| ------------------------------------------------------------------------ | ---------------------------------- |
| train samples                                                            | 57,100                             |
| unique user inputs                                                       | **812**                            |
| unique assistant outputs                                                 | **11,250 (19.7%)**                 |
| unique (input, output) pairs                                             | 15,713                             |
| train samples whose exact output occurs elsewhere in train               | 84.0%                              |
| reply sentences (occurrences / unique)                                   | 129,736 / 4,522 (mean reuse 28.7x) |
| share of sentence occurrences covered by the 1,000 most common sentences | 80.6%                              |

The generator's docstring says "most samples are unique even at 60K scale". **False.**
The generator's own summary line, when re-run, prints `11611 unique outputs, 19.4% unique`.

**Why: 44 of the 60 topics have exactly 10 possible replies.** Those topics are built by
`_topic(user_msgs, guppy_templates, category)`, whose template list is a list literal of
f-strings. The `{pick(...)}` inside them runs **once, at import time**, not per sample.
So each of these topics is a fixed table of 4-5 prompts x 10 replies (20 for
`bubbles`), sampled ~950 times. Every (prompt, reply) pair in those topics occurs about
20 times in train. Only the 15 original hand-built topics (`greeting`, `feeling`,
`food`, ...) compose per call; they have 435 to 836 unique outputs each.

Top train outputs (each repeated ~120 times):

```
132  the glass wall has nice colors. better than me.
122  you can't swim. that must be hard.
121  i don't have seasons in here. just water and sometimes food.
```

### Train/eval overlap (measured)

| eval sample property                                            | rate       |
| --------------------------------------------------------------- | ---------- |
| user input appears verbatim in train                            | **100.0%** |
| exact (input, output) pair appears in train                     | **77.3%**  |
| exact output appears in train                                   | 84.6%      |
| every sentence of the output appears in train                   | 98.8%      |
| output within difflib ratio 0.9 of a same-category train output | 92.6%      |

Near-duplicate example: eval `you put a decoration in here. i'm going to swim around it 30 times.`
vs train `... 20 times.`

So the 3,000-sample eval set is **not held out** in any meaningful sense: 77% of it is
literally in train. Eval loss measures fit to the generator's distribution, which for
this data is nearly the same as fit to the train set.

---

## 2. Memorization in the baseline (measured, then inferred)

`tools/memorization.py` samples 3 replies per prompt for five prompt sets:

- `eval_cases`: the 16 hand-written harness prompts.
- `eval_jsonl`: 60 distinct inputs from `eval.jsonl` (all are train inputs too).
- `novel`: 30 hand-written paraphrases, each checked to be absent from train inputs.
  My first draft had 7 that were already train inputs (`do you like music`,
  `how old are you`, `tell me a joke`, ...); with 812 inputs, ordinary phrasings collide.
- `copy_probe`: "confused" questions about 10 human things **not** in the generator's
  `HUMAN_THINGS` list (`quantum physics`, `a toaster`, `tiktok`, ...). In train, the
  confused reply always names the thing from the question, so copying it is the one
  piece of real composition the data teaches.
- `copy_known`: the same templates with 10 things that **are** in `HUMAN_THINGS`.

Reply classes: **verbatim** = equals some train output; **composed** = not verbatim but
every sentence is a train sentence; **novel sentence** = at least one sentence never seen
in train. `on-topic` uses a classifier that maps each reply sentence to the train
categories it occurs in (heuristic; the `novel` set's acceptable categories are my labels).

Baseline `best_model.pt`:

| set                         | verbatim | composed | has new sentence | sentences seen in train | on-topic | names the thing |
| --------------------------- | -------- | -------- | ---------------- | ----------------------- | -------- | --------------- |
| eval_jsonl                  | 0.68     | 0.29     | 0.03             | 0.99                    | -        | -               |
| eval_cases                  | 0.25     | 0.69     | 0.06             | 0.98                    | 0.79     | -               |
| novel                       | 0.54     | 0.21     | 0.24             | 0.89                    | 0.42     | -               |
| copy_probe (unknown things) | 0.24     | 0.34     | 0.41             | 0.81                    | 0.92     | **0.00**        |
| copy_known (listed things)  | 0.43     | 0.50     | 0.07             | 0.97                    | 0.98     | 0.56            |

What this shows:

- **Sentences are memorized; combinations are composed (in the generative topics).**
  98-99% of sentences the model emits on in-distribution prompts exist verbatim in train.
  But in the 15 per-call-composed topics, most replies are new _combinations_ of those
  sentences, which is what the generator itself does. Examples of composed replies:

  > `hi guppy` -> `hi there. i was just swimming in circles.`
  > `goodnight guppy` -> `goodnight. my fins are already slowing down. maybe tomorrow there will be daphnia.`

  Examples of verbatim recall (fixed-table topics):

  > `do you remember yesterday` -> `i forget some things. but not food. never food.` (train output)
  > `are you afraid of anything` -> `nets. i've never been in one but i know they're bad.` (train output)

- **The model never copies a new word from the prompt. 0 of 90.** It fills the slot from
  its memorized list of 31 things instead:

  > `do you know what basketball is` -> `i don't know what reading is. i only know fish things.`
  > `explain tiktok` -> `i don't know what school is. i only know fish things.`
  > `what do you think about the olympics` -> `is the stock market something that lives in water. i am a fish.`

  For things that _are_ in the list it names the right one only 56% of the time
  (`explain wifi` -> `i have no idea what bluetooth means.`). Inferred: the "copy" is a
  lookup keyed on known tokens, not a copy mechanism.

- **Unknown prompts fall into the "confused" template.** 46% of novel-prompt replies are
  confused-mode, even for prompts squarely about fish life:

  > `do you want a snack` -> `i don't know what social media is. can you explain it in terms of water or food.`
  > `i switched the lamp on` -> `the stock market is beyond me. that is too complex for my small brain.`
  > `the house is freezing tonight` -> `i don't know what taxes. i only know fish things.`

  The persona's "i don't understand human things" is the model's out-of-distribution
  fallback. It reads as in character, which hides how often it fires.

- **Where it does make new sentences, they are slot-filling errors**, not new ideas:
  `is a much something that lives in water.`, `i don't understand a stare at it.`,
  `i don't know what what taxes is.` One genuine-looking new composition:

  > `do you ever wish you had company` -> `i have the small rock and the thermometer. that's enough. i talk to my little body sometimes. they listen. i'll be fine by the filter.`

**Verdict (inferred):** a lookup table with fins, plus a sentence shuffler. It reproduces
the generator's grammar (pick a sentence from slot A, a sentence from slot B), which is
real learned structure, but at the level of whole memorized sentences. It does not copy,
paraphrase, or route unfamiliar wording to the right topic.

---

## 3. Data diet (measured)

Same model, same 10k steps, batch 32. Subsets are seeded random samples of train.

| run      | train ex. | epochs | unique inputs in subset | unique outputs in subset | eval pairs found verbatim in subset |
| -------- | --------- | ------ | ----------------------- | ------------------------ | ----------------------------------- |
| data1k   | 1,000     | 320    | 346                     | 636                      | 24.6%                               |
| data5k   | 5,000     | 64     | 501                     | 1,633                    | 65.3%                               |
| data15k  | 15,000    | 21.3   | 633                     | 3,750                    | 75.6%                               |
| baseline | 57,100    | 5.6    | 812                     | 11,250                   | 77.3%                               |

Loss (full 3,000-sample eval set; train loss is on the subset, eval mode, 100 batches):

| run      | ckpt  | step  | eval loss | train loss | gap   | keyword hit, 480 samples |
| -------- | ----- | ----- | --------- | ---------- | ----- | ------------------------ |
| baseline | best  | 9800  | 0.385     | 0.379      | 0.006 | 0.842 +- 0.017           |
| baseline | final | 10000 | 0.385     | 0.379      | 0.006 | 0.835 +- 0.017           |
| data15k  | best  | 5600  | 0.417     | 0.371      | 0.046 | 0.746 +- 0.020           |
| data15k  | final | 10000 | 0.429     | 0.335      | 0.094 | 0.777 +- 0.019           |
| data5k   | best  | 2800  | 0.523     | 0.356      | 0.167 | 0.821 +- 0.018           |
| data5k   | final | 10000 | 0.564     | 0.302      | 0.262 | 0.867 +- 0.016           |
| data1k   | best  | 800   | 1.546     | 0.372      | 1.17  | 0.540 +- 0.023           |
| data1k   | final | 10000 | 1.816     | 0.256      | 1.56  | 0.650 +- 0.022           |

Eval-loss curve (eval loss at step; from `train.eval_history`, 50-batch eval):

| step    | 400   | 800       | 1600  | 2800      | 4000  | 5600      | 8000  | 9800  |
| ------- | ----- | --------- | ----- | --------- | ----- | --------- | ----- | ----- |
| data1k  | 1.967 | **1.583** | 1.639 | 1.755     | 1.763 | 1.783     | 1.850 | 1.856 |
| data5k  | 1.772 | 0.865     | 0.569 | **0.535** | 0.551 | 0.565     | 0.572 | 0.581 |
| data15k | 1.729 | 0.806     | 0.493 | 0.446     | 0.432 | **0.423** | 0.428 | 0.433 |

**Where overfitting starts:** 1K turns at step 800 (~26 epochs), 5K at step ~2,800
(~18 epochs), 15K at ~5,600 (~12 epochs). The baseline (5.6 epochs) never overfits:
train and eval loss are equal to two decimals, which is expected when eval is a resample
of the same small distribution.

### Memorization vs data size (`final_model.pt`, verbatim measured against the subset each run saw)

| run      | eval_jsonl verbatim | same-input verbatim | distinct replies (eval_jsonl) | novel: on-topic | novel: confused-mode | copy_known names thing |
| -------- | ------------------- | ------------------- | ----------------------------- | --------------- | -------------------- | ---------------------- |
| baseline | 0.69                | 0.49                | 0.92                          | 0.44            | 0.50                 | 0.63                   |
| data15k  | 0.82                | 0.61                | 0.89                          | 0.52            | 0.04                 | 0.71                   |
| data5k   | **0.98**            | **0.92**            | 0.74                          | 0.54            | 0.09                 | 0.42                   |
| data1k   | 0.82                | 0.72                | 0.59                          | 0.29            | 0.07                 | 0.19                   |

Full per-set tables are in `runs/analysis/tables.md` (gitignored); the
`copy_probe` "names the new thing" rate is **0.00 for every model, both checkpoints**.

### Would a person notice the 5K fish? (quoted, final checkpoints, 3 samples each)

Yes, in one way: **it repeats itself.** On in-distribution prompts the 5K fish returns
one memorized line, word for word, sample after sample:

> `want some food?` 5K: `my whole body just got excited. give me the tiny pellets. i promise to eat all of it.` (3 of 3 identical)
> `what do you think about politics` 5K and 15K: `is politics something that lives in water. sounds complicated. the water is nice though.` (3 of 3)
> baseline: `food. did you say food. i like the freeze dried worms best. i'll eat it behind the castle.` / `yes please. i will swim to the top right now. ...` / `yes please. food is my favorite thing after water.`

98% of the 5K fish's replies to eval inputs are verbatim train outputs, 92% are a reply
it was trained on _for that exact prompt_. That is a lookup table in the literal sense.
On the keyword metric it scores as well as or better than baseline (0.867 vs 0.842),
because a memorized on-topic line always hits the keywords.

On novel prompts the smaller fish is arguably _better_: it falls into confused-mode far
less (9% vs 50%), and sometimes stays on topic where baseline deflects:

> `it's sweltering in here` baseline: `inside the second market. i only know fish things.`
> 5K: `warm water is bad for me. i am small and i'll stay in my favorite spot for now.`

But it also produces more broken sentences (`fros live in the water and also outside it. they can't decide. water. i decidided.`).
The 15K fish emitted outright gibberish on one novel prompt
(`do you enjoy songs` -> `i bright ladightuayms. i havezdle through the feeo et.`).

The 1K fish is obviously worse: it collapses to a few stock lines (`i'm slightly different.`,
`i'm famous.`, `hi. i'm where the current is gentle.` 3 of 3), and its "best" checkpoint
(step 800, chosen by eval loss) is still half-trained word salad
(`hi. the fish is my gills feel fine.`). Inferred: on this data, early stopping on eval
loss picks an undertrained model when the set is tiny, because eval loss rises as soon as
the model starts memorizing, which is also when it starts producing fluent text.

---

## 4. Odd things in the data and harness

1. **Eval leakage.** 100% of eval inputs and 77.3% of eval (input, output) pairs are in
   train (section 1). Eval loss can't distinguish generalization from memorization. A
   held-out split should be by input template or by topic, not a random 5% of samples.
2. **Import-time `pick()`** makes 44 of 60 topics fixed 10-reply tables (section 1).
   The f-prefix fix in `5cd3d2f` cured the literal `{pick(...)}` text but left this:
   each slot is still filled once per process. Fix: make the templates lambdas, or build
   them inside `gen()`.
3. **The "unique" claim** in the generator docstring is wrong (19.4% unique outputs by
   its own printout).
4. **`train.jsonl` has 100 extra lines** (57,100) vs `train_openai.jsonl` (57,000). The
   last 100 rows are hand-written, categories `tankmates`, `other_animals`,
   `fish_facts`, `big_water` (25 each), with no generator code in the repo. Regenerating
   with `generate_dataset(60000)` reproduces the first 57,000 train lines and all of
   `eval.jsonl` byte for byte. Provenance of the 100 unknown.
5. **Keyword hit rate is weak.**
   - Matching is substring: `"hi"` matches `nothing`/`this`, `"ok"` matches `look`,
     `"eat"` matches `great`.
   - A prompt-blind model that emits a random train reply scores **0.25**.
   - A pure lookup table that knows each eval case's category and emits a random train
     reply from it scores **0.81**, above the baseline's reported 0.775.
   - With 80 samples (16 prompts x 5) its standard error is about 4.6 points; the baseline
     scored 0.775 with the harness seed and 0.842 with 480 samples. Run-to-run
     differences under ~7 points are noise. Prompt-level variance (only 16 prompts) is
     not even captured by that error bar: the non-monotonic 1K < 15K < 5K ordering is
     probably that.
6. **Eval cases outside the training distribution.** `follow_up` (`thank you guppy`) is
   not a train category, and `what's 2 plus 2`, `thank you guppy`,
   `how are you feeling today`, `want some food?`, `it's really hot today`,
   `i just changed your water`, `sorry i dropped something` are not train inputs. The
   other 9 are verbatim train inputs (`what's on your mind` occurs 100 times). The
   baseline answers `thank you guppy` with `i don't know what cryptocurrency is. if not i probably don't know it.`
7. **The benchmark scores `best_model.pt`**, selected on a 50-batch eval. When eval ~
   train (baseline) this is harmless; for small-data runs it hides the memorized final
   model and picks a different, sometimes undertrained, one. The harness now reports
   final-checkpoint eval and train loss too.

## Measured vs inferred

- Measured: every number in the tables; the import-time `pick()` behaviour (verified by
  counting unique outputs per category and by regenerating the data); overlap rates;
  verbatim/composed/copy rates; loss curves.
- Inferred: that the model "composes sentences but does not copy"; that confused-mode is
  an OOD fallback; that small-set early stopping picks undertrained models; that the
  non-monotonic keyword scores are prompt-level noise. The `on-topic` classifier and
  the `novel` prompts' acceptable categories are heuristic and hand-labelled, and
  novel-set rates rest on 90 replies each, so differences under ~10 points there are
  not meaningful.
