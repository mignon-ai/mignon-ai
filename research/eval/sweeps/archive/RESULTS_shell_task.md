# Second task: shell commands from tldr-pages

`research/tools/build_shell_sft.py --zip tldr.zip` builds a shell-command assistant task (SFT pairs, held-out
pairs and `shell_tasks.json` probes) from tldr-pages, skipping git. Probes are split by
"seen intent" (a trained command with new values and phrasing), "unseen intent" (another example of
a trained tool) and "unseen tool". All runs: `base` preset, pretrained on the tech/git text, SFT
from that checkpoint, one seed, exact match on the first probe of each kind.

## Broad: 3,867 tools, 59k pairs (4 variants per example), 3 epochs

| Probe | Shell only | Git + shell | Git + shell, constrained |
|---|---|---|---|
| Full command, seen intent | 1.7% | 5.0% | 0.0% |
| Option, seen intent | 3.3% | 5.0% | 21.7% |
| Argument copy, seen intent | 33.3% | 35.0% | 46.7% |
| Any unseen intent / unseen tool | 0-7% | 0-7% | 0-15% |
| Overall (558 probes) | 12.0% | 14.5% | 17.9% |

Git exact match with the joint model: 61.6% plain, 63.9% constrained, so adding the shell data does
not hurt the git task (it was 63.5% constrained with git only).

## Narrow: 52 core Unix tools, 10k pairs (40 variants per example), 15 epochs

| Probe | Plain | Constrained |
|---|---|---|
| Full command, seen intent (36) | 61.1% | 41.7% |
| Option, seen intent (28) | 75.0% | 82.1% |
| Argument copy, seen intent (26) | 96.2% | 84.6% |
| Full command / option, unseen intent (60) | 3.3% / 5.0% | 0.0% / 8.3% |
| Argument copy, unseen intent (60) | 40.0% | 40.0% |

## Findings

- The same model that gets 60% on git gets 1-5% on seen-intent full commands over 3,867 tools. The
  limit is data per fact: tldr has about ten commands per tool and each appears 4 times, while git
  has a few hundred command forms repeated thousands of times. A 538k-parameter model cannot store
  tens of thousands of distinct commands.
- Restricting to a narrow domain with many variants per command (52 tools, 40 variants) brings
  seen-intent full commands to 61%, options to 75% and argument copying to 96%.
- Unseen intents stay at 0-5% in both settings, as for unseen git flags. Top-5 for the option is
  95-100%, so the model knows which options are plausible but cannot pick the right one.
- Constrained decoding helps in the broad setting (the grammar is large and the model is unsure)
  and hurts in the narrow one (the grammar learned from 260 forms is too tight for new phrasing).
- Training 15 epochs on the small set erased the general technical-English cloze (0/18).

Conclusion: the model is a specialist. A new domain needs a narrow command set and many varied
requests per command, as for git; a broad man-page-style assistant is out of reach at this size.

## Unix assistant: 73 core tools, tldr plus a hand-written glossary

`research/tools/unix_glossary.py` adds 228 command forms with 2-3 phrasings each, rendered 12 times with varied
values (1 form in 6 is held out whole, as "unseen intent"). Training data: tldr pages of 73 tools
(40 variants per example) plus the glossary, 29k pairs; `research/eval/unix_tasks.json` has 470 probes.

```
python3 research/tools/build_shell_sft.py --zip tldr.zip --out-dir data --tasks research/eval/unix_tasks.json \
  --variants 40 --glossary-repeat 3 --per-category 80 --only ls,cd,cp,mv,rm,mkdir,...
```

`base` preset, pretrained on the tech/git text, one seed, plain decoding unless noted:

| Probe | Unix only, 5 ep | Unix only, 15 ep | Git + unix x4, 3 ep | same, constrained |
|---|---|---|---|---|
| Full command, seen intent (80) | 26.2% | 46.2% | 45.0% | 25.0% |
| Option, seen intent (66) | 56.1% | 68.2% | 65.2% | 68.2% |
| Argument copy, seen intent (66) | 80.3% | 92.4% | 86.4% | 83.3% |
| Full command, unseen intent (80) | 3.8% | 3.8% | 3.8% | 0.0% |
| Argument copy, unseen intent (80) | 43.8% | 47.5% | 56.2% | 50.0% |
| Overall (470) | 32.6% | 41.1% | 40.2% | 35.1% |

The joint model keeps git at 64.2% exact constrained (438 probes), against 63.5% for git only.

- Epochs matter more than data volume: 5 to 15 epochs doubled seen-intent full commands (26% to 46%).
  Mixing in 64k git pairs and repeating the unix pairs 4 times matches the unix-only model.
- Constrained decoding hurts full commands again (45% to 25%): the grammar is learned from replies
  whose arguments vary too much, so valid new combinations are rejected. It does not help here.
- Unseen intents are still near zero (4% full command), the same limit as git.
- The unix-only model forgets the technical-English cloze (0/18); the joint model keeps part of it.

### Failure analysis and the fix

Of the 44 wrong seen-intent full commands, most were the same command in a different spelling
(`free -h` vs `free --human`, `id -u` vs `id --user`): the builder picked the short or long form of
each tldr option at random, so one request had two valid commands. It now picks one form per option
(hash of the placeholder). The other failures are tldr requests that are themselves ambiguous. The
text after a command in the verbose output (`...backup.pdfgit fetch origin`) is not a stopping
problem: the benchmark decodes as many tokens as the expected reply plus <EOS> and does not cut at
<EOS>, so a wrong, shorter command shows the model's continuation after it.

Retrained with the fixed data (git + unix x4, plain decoding):

| Probe | 3 epochs | 6 epochs | INT8 of 6 epochs |
|---|---|---|---|
| Full command, seen intent | 70.0% | 67.5% | 66.2% |
| Option, seen intent | 93.9% | 93.9% | |
| Argument copy, seen intent | 90.9% | 89.4% | |
| Full command, unseen intent | 5.0% | 7.5% | |
| Overall (470) | 49.6% | 49.6% | 49.4% | 
| Git benchmark, constrained (438) | 62.6% | 65.1% | 64.8% |

`models/assistant_base_sft_int8.json` is the 6-epoch model quantized.
