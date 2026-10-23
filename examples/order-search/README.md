# Order search demo

A 538k-parameter model that runs in the browser (WebAssembly through `packages/mignon-web`, INT8, about 15–80 ms per request) and
turns a plain-English request into a search filter for a table of orders:

```
unpaid electronics orders over $500 from last month
  -> orders find --status unpaid --category electronics --min-amount 500 --when last-month
```

The model only writes that command. All task logic is JavaScript in this folder; the engine side is the generic
`@mignon-ai/mignon-web` package (`packages/mignon-web`), which knows nothing about orders.

## Run it

```bash
examples/order-search/build.sh                 # builds the mignon-web package and copies mignon.js and the wasm here
python3 -m http.server -d examples/order-search 8000
# open http://localhost:8000   (?q=unpaid%20orders asks on load)
```

`orders.mgn` (0.6 MB) is the model the page downloads, an INT8 model in the binary model format. `build.sh path/to/model.mgn` replaces it; `mignon convert` rewrites an older JSON model file in this format.

## What is here

| File | Role |
|---|---|
| `index.html` | The page: input, examples, the 5,000 made-up sample orders (generated in the browser), the result, the parsed filters, and the raw reply shaded by how sure the model was of each piece |
| `orders.js` | Everything that knows what an order query is: validates the reply, resolves dates, corrects customer names, computes confidence. Works in the page (`Orders`) and in Node |
| `orders.test.js` | Unit tests for `orders.js` (`node --test examples/order-search/orders.test.js`) |
| `check.js` | Runs the real pipeline (wasm model + `orders.js`) in Node and scores it on `handwritten.json` |
| `handwritten.json` | 168 requests with the expected reply, written by hand (search-box phrases, odd numbers, dates, ids, before/after, names, not-searches), including the ones typed into the page that failed |
| `build.sh` | Builds the `mignon-web` package and copies `mignon.js` and `mignon-web.wasm` next to the page |

## The reply format

`orders <find|count|total> [options]`, options always in this order, or `orders unknown` for anything that is
not an order search (small talk, other topics, actions such as "delete order 5").

| Option | Meaning |
|---|---|
| `--status` | paid, unpaid, overdue, refunded, cancelled, shipped, pending |
| `--category` | electronics, books, clothing, home, toys, sports, beauty, groceries |
| `--min-amount N`, `--max-amount N` | whole numbers; "over", "under", "between", `>`, `$1,350`, "2k" are all written as digits |
| `--when` | today, yesterday, this-week, last-week, this-month, last-month, this-year, last-year |
| `--last-days N` | "the past 30 days" |
| `--month M [--day D] [--year Y]`, `--year Y` | a month, a calendar day or a year |
| `--after X`, `--before X` | open date ranges; X is `october-15`, `october-15-2025`, `march`, `march-2024` or `2024` |
| `--id N` | one order; stands alone |
| `--customer NAME` | quoted when it has spaces |
| `--sort newest|oldest|highest|lowest`, `--limit N` | `find` only |

Dates are symbolic on purpose. The model never does calendar arithmetic: `orders.js` resolves them against
today's date (UTC; weeks run Monday to Sunday; a month or day without a year is its most recent occurrence).
A reply that breaks the format (unknown option, bad value, `--min-amount` above `--max-amount`, two kinds of
date range) is rejected by `orders.js` and the page runs nothing.

## Confidence and corrections

- **Confidence.** Decoding is greedy and every token comes with its probability. The page shades the raw reply
  by it and warns when any piece is below 60%. This catches most wrong guesses, but not all: a model can be
  99% sure of a wrong answer.
- **Customer names.** The model writes a name token by token, so long unusual names drift ("Periklis" came out
  as "Perphlen"). `orders.js` checks that the name appears in the request as whole words; if not, it replaces
  it with the closest run of words from the request and tells the user. The weak pieces of the corrected name
  no longer count against the reply.

## How the model was made

1. **Pretrain** a `base` preset model on general English text (`mignon train -preset base -text ...`; this one
   saw the corpus three times).
2. **Training pairs** from `tools/build_query_sft.py`: a generator that writes a request and its exact reply
   from a random structured query. Fixed option order, arbitrary numbers (a short fixed list made the model
   memorise it and answer 700 for 800), many ways to write dates and amounts, names of many origins, search-box
   style requests (`"unpaid"`, `"over 500"`, `"oct 15"`, `"customer Globex"`, `"all"`), order ids, before/after
   dates, lower-case typing, and `orders unknown` examples (off-topic messages written by a local LLM through
   `tools/generate_off_topic.py`). Some (status, category) combinations are held back to test generalization.
3. **Paraphrases** from `tools/generate_paraphrases.py`: a local Ollama model (qwen2.5:3b-instruct) rewrites
   requests while the reply is copied from the source pair, so the LLM cannot introduce a wrong command. A
   paraphrase is rejected if it drops a number or a name.
4. **Fine-tune** on the pairs (60k generated plus the paraphrases twice: about 73k) and quantize:

```bash
python3 tools/build_query_sft.py --out-dir data/query --train 60000 --off-topic data/off_topic.json
mignon train -load models/pre_base.json -sft data/query/query_sft.json -epochs 3 -lr 0.016 -seed 1 -save models/orders.json
mignon quantize -load models/orders.json -save examples/order-search/orders.mgn
```

A run takes a few minutes on a desktop CPU. The generator also writes `query_heldout.json` and
`query_tasks.json` for `mignon eval -benchmark -tasks ...`.

## How well does it work

Measured on the page's pipeline (`node examples/order-search/check.js`), wasm INT8 model, one training run:

| Set | Result |
|---|---|
| `handwritten.json`, 168 requests | 165 correct (98.2%) |
| generator benchmark, 720 held-out probes | 98.1% (unseen status/category combinations 88.8% full reply, off-topic 96.2%) |
| the 7 requests that failed on the first page version | 6 of 7 correct; "the 5 least expensive orders from the past 30 days" gets `--max-amount 5` wrong, with the weakest piece at 39% (the page warns) |

Read these numbers with care:

- `handwritten.json` is not a clean test. Its requests were written by the person and the assistant building
  this, and the generator was then extended after the first failures, so some wording patterns are now in the
  training data. A fair measure needs requests nobody has seen: log what real users type, label the low-confidence
  ones, and add them to training and to this file.
- Single training runs differ by a few requests (the same data gave 91–100% on the same set across runs), so
  one request more or less is noise.
- Known gaps: a bare single word such as `Globex` or `Peri` is read as "not an order search" (it could be
  anything; `customer Globex` works); "5 priciest unpaid orders" sometimes drops the limit; a numeric id
  inside a longer sentence ("what was ordered on March 3rd") can be read as an id; there is no multi-turn
  context ("and the unpaid ones?"); unseen kinds of filter (a product name, a region) are not in the format at all.
- The failures were coverage gaps, not model size. `orders.mgn` was trained before the generator learned typos, status
  nouns ("refund count"), "last 3 orders", many more refusals and a much wider vocabulary; retrained on that data the same
  `base` model scores ~20 points higher on requests built around those failures (`research/FINDINGS.md`, test sets in
  `research/eval/order/`). Shipping a retrained model is the obvious next step. New kinds of request still mean new
  generator templates and a retrain, which is cheap.
