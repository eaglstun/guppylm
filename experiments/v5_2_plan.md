# Aquarium v5.2 plan (draft)

Implementation status and measured screening results: [v5.2 report](v52/README.md).
The plan below remains the full experiment design; the report distinguishes executed work from pending gates.

## Goal and starting point

Improve replies to new names, nouns, phrasings, and longer conversations without losing the fish's established voice. Keep the 8,873,472-parameter architecture for the first comparison so changes in data and training can be attributed. Start from the completed `aq-td/best.pt` (TinyStories pretraining, then 2,000 steps of TinyDialogues plus ocean dialogue, with a 15% story mix). TinyDialogues is the dialogue source; TinyStories is the earlier pretraining source and retained story mix. Compare against the completed 15,000-step v5 and v5.1 checkpoints, not the interrupted v5.1 run.

The initial deliverable is a reproducible float checkpoint and evaluation. Select its compressed representation **after** training: codec rankings changed between v5 and v5.1. A smaller network can be a later, separately controlled experiment once the data and evaluation are credible.

## What the experiments say

| Completed float run | Chat validation loss | Held-out persona loss | Name | Held-out persona | Noun copy | Memory | Unseen noun/phrasing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v5, two noun-swap copies | 1.7542 | 2.0669 | 1.000 | .625 | .558 | .875 | 6/40 |
| v5.1, four noun-swap copies | 1.7549 | 2.0691 | .979 | .750 | .635 | .875 | 9/40 |

Only the noun-swap count changed between these two completed runs. v5.1 improved the sampled copy/persona probes but not validation loss; 40 unseen cases and one sampled answer per case are too small to establish generalization. Its real-noun copy score rose from .484 to .641 while invented-noun copy fell from .675 to .625. Do not infer that adding more identical swaps is the next improvement.

Short v5 continuation with explicit noun-copy conversations improved the standard probe, but two of its five training question forms matched that probe. The varied continuation reached 16/40 unseen prompts and hurt name and persona scores. This points to a **balanced, varied data intervention** and independent prompt forms, rather than an unqualified copy-data increase. The original template-data experiment found 77.3% exact train/eval input-output overlap; its lesson is about split design, not directly about Aquarium's loss. The original size and persona experiments also used a different model/data regime and do not justify shrinking Aquarium yet.

Compression measurements are checkpoint-specific. On v5.1, tuned d4/4096 scored 2.0897 held-out loss at an estimated 3,568,128 bytes; tuned E8/g128 scored 2.2040 at 2,549,760 tensor bytes; a WebP/E8 hybrid scored 2.1066 at 3,465,162 logical payload bytes (3,491,640-byte ZIP container). Their probe results differ and only E8 has a direct-code browser reader. The same v5.1 E8 browser binary is 2,590,080 bytes and matched Python logits within 1.62e-5; same-run Node generation was 196.6 versus 220.1 tokens/s for int8. Phone performance remains unmeasured. The 1-bit, cellular, layer-pruning, entropy-only, and image-codec-only routes did not offer a compelling measured quality/size tradeoff. See [quantization](../QUANTIZATION.md), [binary confirmation](compression/one_bit_v51.md), [cellular](compression/cellular.md), [data](data.md), [size](size.md), and [persona](persona.md).

## Phase 1: freeze an evaluation contract before generating data

1. Record hashes, row counts, source/generator versions, tokenizer, and split membership for v5/v5.1 inputs. Audit exact and near-duplicate dialogue turns, question templates, names, nouns, and response fragments across all sources, including song and family extras and the 300-dialogue `dialogues_eval.jsonl`. Report source proportions after the 20,000-base-row cap and after augmentation.
2. Create **train**, **development**, and **locked test** manifests. Split source conversations before any rename/noun/memory augmentation; derived rows stay with their parent. Keep the existing 300-dialogue evaluation as a historical comparison/development set because it has guided many codec and training choices. Do not use it as the new locked test. Exclude locked-test entities, wording families, and near duplicates from training and development.
3. Make distinct test slices: seen species with unseen phrasings; held-out species/personas; unseen real and invented nouns; unseen names; multi-turn reference/memory; song/family handling; and free-form conversation/voice. Keep a small, fixed representative story sample to detect regression from the TinyStories capability. Publish the prompt list, decoding settings, scoring rubric, and confidence intervals before comparing candidates. Use several samples per prompt and substantially more than the existing 40-case noun set.
4. Measure assistant-token loss per source and slice as well as sampled behavior. Include prompt-blind/lookup-style baselines where applicable. Human-review a blind sample for coherent, in-character replies; exact-word probes alone miss that quality. Check each evaluation prompt against all training strings and generated templates.

## Phase 2: data and training ablations

Use the same `aq-td/best.pt`, tokenizer, architecture, 15,000-step budget, batch/sequence length, 15% story mix, and base dialogue cap for initial matched runs. Hold a separate seed for a confirmation run. Record effective examples/tokens by source, not only row counts.

| Candidate | Change from v5 recipe | Question |
| --- | --- | --- |
| A, reproduction | Existing two noun swaps, new split only | Does the corrected validation contract change the apparent baseline? |
| B, v5.1 reproduction | Four noun swaps, new split only | Does its advantage persist on independent prompts? |
| C, varied-copy mix | Two swaps plus a bounded share of new, diverse noun/name reference conversations, with wording families held out | Can copy improve without persona/name loss? |
| D, balance adjustment | Best of A-C with source and skill sampling adjusted from the audit | Do song/family/persona/story or repeated templates dominate the batch? |

Prefer generating new conversations with varied syntax, roles, turn positions, and response patterns to making more substitutions of the same template. Include instructions that require copying an unfamiliar token, declining to invent memory, and maintaining the named creature's voice in the same conversation. Inspect and reject bad generations. Keep a manifest of source prompt, generator, ID, parent ID, and split. Deduplicate before training; never move a row across splits to improve a score. Test a short, low-learning-rate continuation from both v5 and v5.1 only as a cheaper screening experiment; compare it separately from equal-budget full runs.

Update the trainer to accept explicit split manifests and log per-source sampling. Save optimizer, scheduler, random states, step, and data-order state with checkpoints so interrupted runs can resume faithfully. Choose checkpoints on the development score/rubric, not on the locked test. Keep the existing historical validation and probe scores in reports for continuity, with clear labels that they were used during prior selection.

## Phase 3: compression on the selected v5.2 float weights

Evaluate int8 per-row as the browser quality reference. Refit d4/4096 rotated GPTQ, E8/g128, and the WebP/E8 hybrid **from v5.2**, using train-only calibration and a separate compression development set. Run a small Pareto sweep of size, assistant-token loss, and each behavior slice; select a checkpoint/codec pair before opening the locked test. Count actual downloadable files and decoder code, peak/resident memory, prompt time, and token rate. For d4, build and measure a real packer before claiming its estimated size as a deployable size. For the hybrid, build a browser reader before comparing user-facing performance. Verify Python↔JavaScript logits, deterministic generation, and real phone WebView performance. If the float model improves but compression loses its gains, ship the float/int8 result as the model milestone and continue codec work separately.

## Decision record and exit criteria

Before training, write a numerical acceptance rubric from the new baseline distribution: minimum acceptable name, persona, copy, memory, story, and human voice scores; maximum allowed loss/latency regression; and the desired download-size tier. Include uncertainty, not just point estimates. Promote v5.2 only if its locked-test quality improves the relevant v5.1 weaknesses without a material regression in the other slices, and the selected browser artifact passes parity and phone checks. If no candidate clears the rubric, keep v5.1 and report the failed ablations.

Archive the manifests, generation prompts, data audit, exact commands/configs, run seeds, checkpoints, probe outputs, codec payloads, and deployment measurements. Record both positive and negative results in this experiment index. No result on the reused 300-dialogue set alone is evidence of a v5.2 improvement.
