# Constrained decoding of git commands

`mignon eval|chat|generate -constrain data/git_sft_v2.json` (`internal/constrain`). The reply is
restricted word by word: the command and subcommand come from the training replies, an option has
to be one seen with that subcommand, and every other word has to be copied from the request or be
answer vocabulary (a non-option word that 0.5% of the training replies use without their request
containing it: origin, HEAD, user.name). Digits are matched exactly for request words and as a
pattern for answer vocabulary (HEAD~3). Replies whose first word is not a command are not touched.

Model: small, hidden 256, SFT on `data/git_sft_v2.json`; `research/eval/git_tasks.json`, 438 tasks, greedy.

| | Exact float | Exact INT8 |
|---|---|---|
| unconstrained | 160 (36.5%) | 164 (37.4%) |
| constrained | 195 (44.5%) | 195 (44.5%) |

| category | unconstrained | constrained |
|---|---|---|
| Git argument copy | 16/60 | 33/60 |
| Git full command (seen intent) | 1/60 | 16/60 |
| Git option (seen intent) | 50/60 | 53/60 |
| everything else | unchanged | unchanged |

Unseen intents (full command 0/60, option 3%) do not move: the constraint removes invented
words, it cannot supply a subcommand or option the model does not know. 82% of the remaining
failures are knowledge failures, so the next lever is data (domain vocabulary, distillation).

What did not work, on the way: allowing every word seen in 5+ replies (the dataset reuses branch
names without the request mentioning them, so invented names passed), digit-normalized matching
for request words (an invented `issue-201-db-pool` matched `issue-2301-db-pool`), and not trimming
the `?` of a request (a copied name could never be completed).

## Glossary data (research/tools/git_glossary.py)

Hand-written flag glossary: 225 command forms, 2-3 phrasings each, 3 name fillings per phrasing.
Pairs whose command skeleton matches an eval command are excluded from training (207 pairs), and
one entry in five is held out (216 pairs). The SFT run adds the training slice x10 (675 -> 6750 pairs
on top of 57k). Same pretrained `small` model, 3 SFT epochs, 3 seeds (the seeds barely differ, so
read it as one run):

| exact match, git_tasks.json | baseline | + glossary |
|---|---|---|
| overall, unconstrained | 36.2% | 39.2% |
| overall, constrained | 43.5% | 46.2% |
| subcommand (unseen intent) | 53% | 67% |
| full command (unseen intent) | 0% | 0% |
| option (unseen intent) | 3.3% | 3.3-4.4% |
| held-out glossary entries (full command) | 0% | 1.4% |

The glossary teaches the model which subcommand a new phrasing means, and nothing about flags it
was not shown: the held-out glossary flags and the benchmark's unseen options stay at about zero.
A 219k model memorizes request-to-flag pairs; it does not infer a flag's meaning.

## Capacity check: `base` preset (538k params) on the same data

Same pretraining text (44.7k seqs, 10 epochs), same SFT (`git_sft_v2`, 3 epochs), 2 seeds (they differ by under 1 point).
A = baseline SFT, B = baseline + glossary x10. Exact match on `research/eval/git_tasks.json`.

| | small A | small B | base A | base B |
|---|---|---|---|---|
| overall, plain | 36.2% | 39.2% | 59.6% | 61.3% |
| overall, constrained | 43.5% | 46.2% | 61.9% | 62.7% |
| argument copy, plain | - | - | 92.5% | 93.3% |
| subcommand (unseen intent), plain | 53% | 67% | 62% | 72.5% |
| option (unseen intent), plain | ~3% | ~3-4% | 2.5% | 4.2% |
| full command (unseen intent), plain | 0% | 0% | 1.7% | 0% |
| held-out glossary flags, plain | 0% | 1.4% | 1.4% | 3.7% |

Findings:
- Going from `small` to `base` is worth about +23 points plain, far more than the glossary (+2 to +3) or constrained decoding.
- The constraint helps much less on `base` (+2 points plain A to constrained A, versus +7 on small): the model already copies arguments (92.5%).
- Unseen-intent flags and full commands are still near zero at 538k params, so those categories are not a capacity problem at this size. They are a coverage problem: the model reproduces only pairs it has seen.
- The glossary still helps subcommand mapping (+10 points).

## Larger glossary (second pass): no further gain

The glossary grew from 225 to 479 command forms (1,839 training pairs, 516 held out, 525 dropped as eval-skeleton overlap).
`base` model, 2 seeds, A = SFT without glossary, C = with the larger glossary x10.

| exact match | A | C |
|---|---|---|
| overall, plain | 59.8-60.0% | 60.0-60.5% |
| overall, constrained | 62.1% | 61.6% |
| subcommand (unseen intent), plain | 61.7% | 65-66.7% |
| option (unseen intent), plain | 1.7-3.3% | 3.3-5.0% |
| held-out glossary flags (new 516) | 0.6% | 1.7-2.3% |

The first glossary (225 forms) gave 61.3% plain on `base`; 2.7x more entries add nothing measurable.
Flag meanings seen in training do not transfer to flags not seen, so more glossary is not the lever.

## Capacity check: `large` preset (1.95M params)

One seed, same pretraining text and SFT as the `base` runs above. A = no glossary, C = larger glossary x10.

| exact match | base A | base C | large A | large C |
|---|---|---|---|---|
| overall, plain | 59.8-60.0% | 60.0-60.5% | 65.5% | 66.4% |
| overall, constrained | 62.1% | 61.6% | 66.7% | 67.8% |
| Top-5 | - | 93.4% | 97.5% | 97.7% |

`small` -> `base` was +23 points; `base` -> `large` is +6 (3.6x the parameters, about 2.6x slower training).
Returns are shrinking, so `base` is the better size/quality trade-off for the git assistant.
