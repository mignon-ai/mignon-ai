# Dilation start and rotary latch (Oct 2026)

Text val bits/byte on a git+tech corpus subset (4 files of pretrain_git_v3, 2 of pretrain_tech_v3),
Muon, 3 epochs, max_seqs 60000, seeds 1-3, paired against the first variant of each sweep.
Specs: dilation_start.json, latch_rotary.json (the latter uses the removed config key latch_rotary;
the rotary latch was removed from the code after this result). The rotary specs ran as separate
plain-vs-rotary sweeps on small, small without attention (attn_heads 0) and micro.

| sweep | first variant | second variant | delta | verdict |
|---|---|---|---|---|
| dilation, micro | 2.7111 | 2.7135 | +0.0025 | same |
| dilation, small | 2.3387 | 2.3459 | +0.0071 | worse |
| dilation, base | 1.9530 | 1.9634 | +0.0105 | noise, worse on 3/3 seeds |
| rotary, small | 2.3383 | 2.3338 | -0.0045 | same |
| rotary, small no attention | 2.3489 | 2.3548 | +0.0059 | worse |
| rotary, micro | 2.7111 | 2.7088 | -0.0023 | same |

Dilation: first variant dilation_start=1 (dilated slot unused), second dilation_start = the dilation base
(0 meant "base" when these ran; 0 now means 1, so the archived spec writes the base explicitly).
Rotary: first variant plain, second variant rotary, both with dilation_start=1.
Rotary latch cost about 1.5% throughput and did not improve any benchmark.

Per-seed logs were not kept; the table above is the paired mean of 3 seeds.

## Neighbourhood ablation: 3-slot vs 4-slot rule (Oct 2026)

Same corpus and settings as above, 5 seeds, paired by seed against `old3`. `old3` is the original
rule `[self | left(dilated) | latch]` (HEAD 676c3d4); `old3_matched` is the same rule with a wider
hidden layer so its parameter count equals the 4-slot model (small: 96 -> 120, micro: 48 -> 60);
`new4_*` is `[self | prev | dilated | latch]`. Values are val bits/byte, lower is better.

| config | old3 | old3_matched | new4 start=1 | new4 start=base |
|---|---|---|---|---|
| micro (4 steps, base 3, no attention) | 2.7258 | **2.7061** (-0.0197 ± 0.0048) | 2.7133 (-0.0126 ± 0.0019) | 2.7163 (-0.0096 ± 0.0024) |
| small (1 step, attention) | 2.3364 | **2.3234** (-0.0130 ± 0.0013) | 2.3373 (+0.0009 ± 0.0021) | n/a (slot unused) |
| small, 3 steps base 2 | 2.3404 | **2.3274** (-0.0130 ± 0.0008) | 2.3383 (-0.0022 ± 0.0020) | 2.3428 (+0.0024 ± 0.0017) |

(± is the standard error of the paired difference.)

Throughput (tok/s): the matched 3-slot model is 8-18% slower than old3 and 2-10% slower than the 4-slot
model (micro 377k vs 420k, small 235k vs 241k, small 3 steps 154k vs 162k), because the extra
parameters sit in a wider hidden layer. The comparison above is at equal epochs, not equal time.

Conclusion: the extra stride-1 tap helps only on micro (-0.013) and is flat elsewhere; spending the
same parameters on hidden width is better in every configuration (-0.013 to -0.020). Starting the
dilation at the base is never better than starting at 1.

Outcome: the rule was reverted to the 3-slot `[self | left | latch]` form and the `dilation_start`
config key, `-dilation-start` flag and `initial_dilation` metadata were removed, so
`dilation_start.json` in this directory no longer resolves (like `latch_rotary.json`). Hidden width
is the better place to spend parameters; retune it per preset with equal-time sweeps.
