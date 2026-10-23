# Hidden width per preset (Oct 2026)

Text val bits/byte on a git+tech corpus subset (4 files of pretrain_git_v3, 2 of pretrain_tech_v3),
Muon at each preset's tuned LR, paired by seed against the preset width. "Equal time" means each
width got a `max_seqs` scaled from its measured throughput so wall time per run matched.
Specs are not kept as files: they were `hidden_dim` variants of the `small` and `base` presets.

## small (1 block, state 64), 3 epochs, equal time, 5 seeds

| hidden | bpb | vs 96 (preset) |
|---|---|---|
| 64 | 2.3603 | +0.0242 |
| 96 (old preset) | 2.3361 / 2.3375 | baseline |
| 128 | 2.3212 | -0.0149 |
| 192 | 2.3095 | -0.0266 |
| **256 (new preset)** | 2.3011 / 2.3017 | -0.0350 / -0.0358 |
| 384 | 2.2963 | -0.0413 ± 0.0028 |
| 512 | 2.3004 | -0.0371 ± 0.0023 |

The knee is at 256 to 384; 384 is not distinguishable from 256 within noise. The preset Muon LR
(0.08) was tuned at hidden 96 and was not re-swept at 256. Training throughput at 256 is about 26%
lower per token than at 96.

## base (2 blocks, state 96), 3 epochs, 3 seeds

| setting | hidden 160 (preset) | 256 | 384 | 512 |
|---|---|---|---|---|
| equal time | 1.9521 | 1.9572 (+0.0051 ± 0.0034) | 1.9899 (+0.0377) | 2.0444 (+0.0923) |
| equal tokens (60k seqs) | 1.9521 | 1.9276 (-0.0245 ± 0.0030) | 1.9131 (-0.0390 ± 0.0033) | not run |

LR at hidden 256, equal tokens: 0.02 -> 1.9273, 0.04 (default) -> 1.9276, 0.08 -> 1.9776.

Reading: at equal tokens wider is clearly better. The equal-time loss at 3 epochs comes from the
wide models seeing 13-27% fewer sequences while the learning curve is still steep, not from the
width or the LR.

## 10 epochs (the presets' real budget), equal time

base, 2 seeds: hidden 160 (preset) 1.9087, 256 1.9209 (+0.0122 ± 0.0019), 384 1.9632 (+0.0545 ± 0.0005).
Longer training widened the gap, so the `base` preset stays at 160.

small, 3 seeds: hidden 96 (old preset) 2.2886, 192 2.2616 (-0.0270 ± 0.0019), 256 2.2556
(-0.0330 ± 0.0025), every seed in the same direction. The `small` preset moved to 256.

The 3-epoch equal-time results overstated the case for width on `base` (+0.005 there, +0.012 at 10
epochs) but held on `small`, so 10-epoch equal-time runs are what decide a preset.

## small, Muon LR at hidden 256 (10 epochs, 3 seeds, equal time)

| LR | bpb | paired Δ |
|---|---|---|
| 0.02 | 2.2516 | +0.0044 ± 0.0043 vs 0.04 |
| 0.03 | 2.2486 | +0.0014 ± 0.0062 vs 0.04 |
| **0.04 (new preset)** | 2.2472 | baseline |
| 0.06 | 2.2511 | -0.0051 ± 0.0007 vs 0.08 |
| 0.08 (old preset) | 2.2563 | -0.0085 ± 0.0012 for 0.04 |

The optimum is bracketed, flat between 0.03 and 0.04 (differences within noise); 0.04 is the
`DefaultMuonLR`. An earlier LR sweep ran at hidden 96 by mistake and is not reported.

## INT8 inference cost of small, 96 → 256

| hidden | µs/token | model file |
|---|---|---|
| 96 | 29.8 | 1.7 MB |
| 256 | 38.4 (+29%) | 3.0 MB |

## INT8 vs float on the task eval (small, hidden 256)

One model (pretrained 10 epochs on a 6-file subset, SFT 3 epochs on `data/git_sft_v2.json`),
`research/eval/git_tasks.json`, 438 tasks:

| | Exact | Top-5 | NLL |
|---|---|---|---|
| float | 160/438 (36.5%) | 83.8% | 1.602 |
| INT8 | 164/438 (37.4%) | 83.3% | 1.595 |

No quantization loss at this size; the per-category differences (up to 3 tasks) go both ways, so
they are noise. QAT is not needed for `small`.

(The Top-5 column of this table first read 17% because the benchmark ranked the first-token logits
after decoding had overwritten the model's output buffer; `internal/eval/suite.go` now ranks them
first. Exact and NLL were not affected.)
