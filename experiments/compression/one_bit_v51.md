# Completed Aquarium v5.1: 1-bit confirmation

The completed `runs/aq-chat-v51/best.pt` checkpoint in the Aquarium worktree
comes from a 15,000-step run (`runs/aq-chat-v51/log.json`). It is separate from
the earlier interrupted `aq-chat-v51-stopped` run mentioned in the older
quantization notes. This confirmation runs the same 1-bit matrix sweep used for
Aquarium v4 against the completed checkpoint.

The score uses the Aquarium evaluator's mean batch cross-entropy on 300 held-out
persona dialogues (28,800 assistant tokens). It is directly comparable within
this table; lower is better. Each 1-bit matrix uses binary signs and an FP16
mean-absolute scale per row. The other matrices retain shipped-style int8
rounding. The tied token embedding/output head is one unique matrix.

| v5.1 variant | Held-out loss | Change from int8 |
| --- | ---: | ---: |
| FP32 | 2.0691 | — |
| All 2D matrices int8 | 2.0688 | reference |
| Tied token embedding alone 1-bit | 3.8111 | +1.7423 |
| All block linears 1-bit, embeddings int8 | 5.4952 | +3.4264 |
| All 2D matrices 1-bit | 6.2417 | +4.1730 |
| Greedy mix: block 3 attention output and FFN down 1-bit | 2.1170 | +0.0482 |

Every individual matrix substitution raised loss. With a +0.05 loss budget,
the greedy mix selected two block 3 matrices. Their estimated packed payload
would save 388,608 bytes, about 4.3% of the 9,042,320-byte v4 web binary.
That percentage is only a format projection: the live page still serves v4,
and no packed v5.1 browser artifact or decoder was built. The greedy search
selected and scored on the same held-out set, so it needs an independent
behavior check before any deployment decision.

The completed v5.1 checkpoint confirms the v4 conclusion: post-training 1-bit
rounding is not a usable whole-model replacement for the Pages demo. A model
trained to tolerate binary weights would be a separate experiment.

## Reproduce

```bash
python tools/experiment_binary_sweep.py \
  --aquarium-root /path/to/guppylm-exp-aquarium \
  --checkpoint /path/to/guppylm-exp-aquarium/runs/aq-chat-v51/best.pt \
  --output runs/binary_sweep_v51/results.json
```

The complete individual-matrix and combination results are in
`results/one_bit_v51.json`.
