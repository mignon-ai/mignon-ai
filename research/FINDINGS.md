# Findings: what limits a small model on a narrow task

Oct 2026, on the order-search model (`examples/order-search`): a plain-English request becomes a command such as
`orders find --status unpaid --min-amount 500 --when last-month`, or `orders unknown` when it is not an order search.
We asked what makes such a model better: more parameters, English pretraining, a better tokenizer, or better training data.

## Short answer

**Training-data coverage first, then English pretraining together with enough size, then the rest.** A 538k-parameter model
went from 75% to 95% on a test set that targets the failures seen, by fixing what the training data lacked. Model size alone, request-style pretraining,
the tokenizer and grammar constraints did not move accuracy.

## Test sets

All are request/reply files scored through the page pipeline (`node examples/order-search/check.js <file>`).

| Set | Size | Notes |
|---|---|---|
| `eval/order/hw5.json` | 88 | Used to find data gaps, so now a dev set |
| `eval/order/hw6.json` | 104 | Written after the failure analysis; saturates at ~95% |
| `eval/order/hw7.json` | 98 | Indirect, wordy, slangy wording; the main discriminating set. Written by the person who wrote the generator's wording, so it flatters later wording changes |
| `eval/order/hw8.json` | 85 | Written by Qwen 2.5 3B from descriptions, then reviewed by hand. Qwen also wrote the training paraphrases, so models trained with them have a style advantage on it |
| "older sets" | 168 | `examples/order-search/handwritten.json`: used for tuning earlier, saturated at 96-99%; a regression check only |

One request is about 1 point and seed spread is 1-4 points, so differences under ~3 points are noise. Every number below is
the mean of 3 seeds unless noted.

## Results

| Question | Comparison | hw6 | hw7 |
|---|---|---|---|
| Training data | old data / generator fixes only / fixes + paraphrases (same 538k base) | 75 / 92 / 95 | 61 / 53 / 67 |
| Tokenizer (English base, 538k) | old 2k / new 2k / 4k / 8k | 96 / 95 / 96 / 96 | 67 / 70 / 69 / 67 |
| Size and English pretraining (old wording) | queries-only 0.5M / 2M | 97 / 97 | 69 / 68 |
| | English 0.5M / **2M** / 5.9M | 96 / 96 / 96 | 67 / **74.5** / 72 (±6.6) |
| Wider wording (v2) | English 2M / queries-only 2M | 94 / 93 | 76 / 74.5 |

## What we learned

1. **Failures were coverage gaps, and they repeated.** The same requests failed in all 15 runs of the first experiment: typos,
   a status noun ("refund count"), a trailing count word, "last 3 orders", refusals for actions ("ship order 90") and for
   imperatives with name-like words ("play some music"). Adding these to the generator was worth ~20 points on hw6, which is built around these classes; the gain on general wording is the smaller one in learning 2.
2. **LLM paraphrases are worth ~14 points on hard wording** (hw7 53% to 67%), even though a 3B model keeps most of the source
   words and sometimes mislabels a request (e.g. "older than 16 days" kept as `--last-days 16`).
3. **A retrained tokenizer is a speed win, not an accuracy win.** Trained on English plus the task text, it cuts replies from
   3.2 to 2.0 tokens per word, makes every task word one token, and improves bits per byte (1.86 to 1.69 at 8k). Accuracy
   is unchanged at this size.
4. **English pretraining needs size.** Neither a bigger queries-only model nor a small English one helped; the 2M English
   model reached 74.5% on hw7 against 68-69%. A 5.9M model was no better than 2M.
5. **Wording diversity then closed most of that gap.** With a wider generator vocabulary (statuses, categories, times, frames),
   a queries-only 2M model rose by 6 points on hw7 and the English advantage fell to ~2 points (noise). English keeps a steady
   ~3-point edge on short, terse requests (98.6% vs 95.0% on the older sets).
6. **Request-style pretraining did not help** (Banking77, CLINC150, MASSIVE phrasings): all variants within noise.
7. **Grammar constraints would not help this task.** About 95% of misses are valid commands with the wrong content.

## Caveats

- Test sets are small and written by us or by Qwen, not by users. No number here measures real-user accuracy.
- One English corpus (Simple Wikipedia, tech text, request phrasings), one tokenizer recipe, no learning-rate tuning at 2M and
  5.9M (rates are the 2M preset's), and one seed for the queries-only 5.9M control.
- The queries-only controls train 10 epochs and the fine-tunes 3, with a task-specific tokenizer for the controls.

## Next

Real user requests as a test set and for training; more varied and checked wording. A bigger model is not the lever.

## Reproduce

The generator is `tools/build_query_sft.py`. The experiment drivers, corpora and models (`data/exp/exp13`..`exp15`) are gitignored
and not part of the repository: each run script is resumable and builds on `mignon train -text` / `-sft`, using the 8k tokenizer
trained on `text_en` plus the task text. The English `large` preset is `large` (2M); `xl` is the 5.9M preset.
