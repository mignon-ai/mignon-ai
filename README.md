# mignon-ai

A small language model that is cheap to train and run on a CPU, in pure Go (no CGO, no Python at run time). It is built for
narrow tasks: turning a request into a command or a structured query, with a model of 0.03M to 6M parameters that trains
in minutes to hours on a desktop CPU and runs INT8 in a browser through WebAssembly. It is not a general chat model.

The model is a **Hybrid Neural Cellular Automaton (Hybrid NCA) with sparse attention**: a local rule network mixes each
token with its causal neighbours and a running prefix memory (selective latch), and an optional windowed attention sublayer
handles associative retrieval (copying, induction, recall). Training uses a multi-threaded CPU engine with AVX2 kernels
(portable Go fallbacks elsewhere) and the Muon optimizer. Inference per-token cost and memory do not grow with the
sequence length.

## Strengths

- **Small and self-contained.** One Go module, no native dependencies. A task model is 0.6 MB (INT8), loads in a page, and
  answers in tens of milliseconds; the engine's WebAssembly build is 5 MB (`packages/mignon-web`).
- **Constant-cost generation.** Windowed attention and a decay memory: memory and time per token stay flat at any length.
- **Reproducible and tested.** A golden snapshot pins model, data and checkpoint behaviour across refactors; gradient checks,
  quantization-equivalence tests, and resumable, seeded training pipelines.
- **A small public API** (`pkg/`) for loading, training, fine-tuning, quantizing, generating with token probabilities, and
  evaluating with calibration. The engine knows nothing about what a prompt means.
- **Language-model quality at tiny size.** At ~1M parameters it matched HuggingFace's `TinyStories-1M` in bits per byte and
  story coherence (earlier work; benchmarks in `research/tools/`).

## Shortcomings

- **Narrow by design.** It handles only what its training data covers. On the order-search task the remaining errors are
  requests worded unlike anything it was trained on; the Unix assistant scores under 8% on commands it never saw.
- **Accuracy depends on the training data.** Coverage and wording variety decide the result far more than parameter count
  (see below). New request kinds need new generator templates and a retrain.
- **No evidence on real users yet.** All test sets were written by us or by a small LLM, and are small (85-104 requests); see
  `research/FINDINGS.md` for what each is worth.
- **Small context.** 128 tokens, with attention over the last 64; no multi-turn memory.
- **Models above ~2M parameters showed no gain** on the task tested, and learning rates are tuned only up to the `large` preset.
- **CPU only.** There is no GPU path; larger models than the presets here would be slow to train.
- **Reproducing the experiments needs local data** (`data/`, gitignored: corpora, a local Ollama model for paraphrases). The git
  history also carries old model files (~100 MB) that were committed before `models/` was ignored.

## What we found

Details, numbers and caveats are in [research/FINDINGS.md](research/FINDINGS.md). In short, on the order-search model:

- Fixing gaps in the training data took a 538k model from 75% to 95% on a test set built around the failures seen.
- LLM paraphrases and wider generator wording each gave the biggest gains on hard, indirect phrasings.
- English pretraining helped only together with enough size (2M), and a queries-only model with wide wording nearly closed that gap.
- A retrained tokenizer shortens replies by a third and models English better, but did not change task accuracy.
- A 5.9M model was no better than 2M; grammar constraints would not help (valid-but-wrong answers dominate).

## Architecture

- **Cells.** Every token is a cell holding a state vector. One shared rule is applied to all
  cells: a gated MLP (GLU) reads `[self ‖ left neighbour ‖ latch]` and a highway gate decides how
  much of its output replaces the state, `s' = (1-g)·s + g·u`.
- **Dilated neighbours.** The left neighbour sits at a stride that grows geometrically with the
  step (`dilation^step`, capped), so a few steps reach far back.
- **Selective latch.** A causal decay memory whose per-channel retention is gated by the current
  state gives every cell a summary of its prefix.
- **Blocks.** One or more blocks are stacked with residual connections between them; each block
  runs a configurable number of rule steps (the larger presets use one).
- **Attention (optional).** `attn_heads` adds a causal multi-head attention sublayer after each
  block, restricted to the last `attn_window` tokens so generation stays constant-cost.
- **Head.** Final RMSNorm and a (by default tied) softmax head over a byte-pair vocabulary.

## Layout

| Package | Purpose |
|---|---|
| `cmd/mignon` | CLI: load, train, evaluate, quantize, save, generate, sweep |
| `internal/ca` | Model definition, parameter layout, dense forward/backward training workspace |
| `internal/nn` | RMSNorm, the attention sublayer and the softmax output head |
| `internal/quant` | INT8 model, streaming generation sessions, quantized checkpoint format |
| `internal/train` | Training loop, configs, presets |
| `internal/optim` | AdamW, Muon, learning-rate schedule |
| `internal/data` | Corpus loading, document splits, sequence encoding, on-disk dataset cache |
| `internal/tokenizer` | BPE tokenizer with chat special tokens |
| `internal/generate`, `internal/infer` | Sampling, chat REPL, generation sessions |
| `internal/constrain` | Command grammar for decoding: restricts replies to request words and learned command vocabulary |
| `internal/eval`, `internal/probe`, `internal/sweep` | Held-out perplexity, task benchmarks, synthetic probes, multi-seed sweeps |
| `internal/kernels`, `internal/parallel` | GEMM, SIMD vector ops, INT8 kernels, work splitting |
| `internal/checkpoint` | Full-precision checkpoint format |
| `internal/golden` | Snapshot test pinning model, data and checkpoint behaviour across refactors |
| `internal/genbench` | Generation cost benchmarks |
| `cmd/mignon-web`, `packages/mignon-web` | The WebAssembly build of inference and the npm package around it (`@mignon-ai/mignon-web`: wasm, a JavaScript API, types; see its README) |
| `examples/order-search` | A browser demo built on that package that turns requests into order filters (see its README) |
| `pkg/model`, `pkg/data`, `pkg/train`, `pkg/infer`, `pkg/eval` | The public Go API: load, save and quantize models, train and fine-tune with cancellation and progress callbacks, generate with token probabilities, evaluate with calibration. `cmd/mignon` is built on it where it can |
| `tools/` | Python scripts that generate prompt/reply training data from structured queries (the order-search generator, local-LLM paraphrases and off-topic messages, request-phrasing downloads) |
| `research/` | [Findings](research/FINDINGS.md), order-search test sets, the git and Unix command assistants, corpus building, TinyStories benchmarks, and the sweep logs that tuned the presets (see its README) |

## Build and test

Requires Go 1.25+.

```bash
go build ./cmd/mignon
go test ./...
go test -bench=. ./internal/genbench -benchtime 3x   # generation cost
```

`MIGNON_PROBE_REGRESSION=1 go test -run TestProbeRegression ./internal/sweep` runs the probe
regression guard against `research/eval/probe_baseline.json` (several minutes).

## Using it as a library

```go
import (
    "github.com/mignon-ai/mignon-ai/pkg/data"
    "github.com/mignon-ai/mignon-ai/pkg/eval"
    "github.com/mignon-ai/mignon-ai/pkg/infer"
    "github.com/mignon-ai/mignon-ai/pkg/model"
    "github.com/mignon-ai/mignon-ai/pkg/train"
)

pairs, _ := data.LoadPairs("pairs.json")          // [{"prompt": ..., "response": ...}]
cfg, _ := train.PresetConfig("base")
m, _, _ := train.FineTune(ctx, pairs, train.Options{Config: cfg, Seed: 1,
    OnEpoch: func(e train.Epoch) { log.Printf("epoch %d/%d val %.3f", e.Epoch, e.Epochs, e.ValLoss) }})
m.SetExtra(json.RawMessage(`{"task":"orders"}`))   // any JSON travels with the model file
q, _ := m.Quantize()
q.Save("model.mgn")                                // binary model file, about 1 byte per INT8 weight

report, _ := eval.Evaluate(ctx, q, heldOutPairs, eval.Options{})  // accuracy and confidence calibration
out, _ := infer.Generate(ctx, q, "unpaid orders", infer.Options{}) // out.Text, out.Tokens[i].Prob
```

The engine knows nothing about what a prompt or a reply means; programs built on it define that.

## Model files

`Save` and every training command write the binary model file: a small header (JSON: shape, metadata, the
tokenizer, and the opaque `extra` field) followed by the weights as raw INT8 or float32 numbers. The layout is
documented in `internal/binfmt`. A model that took 7.1 MB as JSON takes 0.6 MB. The older JSON files still load,
and `mignon convert -load old.json -save new.mgn` rewrites them. Any command that takes `-load` accepts either.

## Usage

```bash
# Pretrain on a directory of .txt files, then chat-tune and quantize
mignon train -preset base -text data/text -save models/base.json
mignon train -load models/base.json -sft data/pairs.json -save models/chat.json
mignon quantize -load models/chat.json -save models/chat_int8.json
mignon chat -load models/chat_int8.json

# One-shot generation, held-out perplexity, benchmark suite
mignon generate -load models/base.json -prompt "git commit" -mode text
mignon eval -load models/base.json -heldout held_out.txt
mignon eval -load models/chat.json -benchmark -tasks research/eval/git_tasks.json

# Restrict command replies to words of the request and of the training replies
mignon generate -load models/chat.json -prompt "squash merge login" -constrain data/git_sft_v2.json
mignon eval -load models/chat.json -benchmark -tasks research/eval/git_tasks.json -constrain data/git_sft_v2.json

# Compare configurations over several seeds
mignon sweep research/eval/sweeps/<spec>.json
```

Run `mignon -h` for the commands and `mignon <command> -h` for a command's flags. `-load` and the
sampling flags (`-mode`, `-temp`, `-top-k`, ...) are shared by the commands that need them. For
`mignon train`, flags set on the command line override the preset.

## Presets

`mignon train -preset <name>` bundles the architecture and the tuned training hyperparameters.

| Preset | Blocks | State | Hidden | Steps | Attention | Context | Vocab | Params |
|---|---|---|---|---|---|---|---|---|
| `micro` | 1 | 32 | 48 | 4 | none | 32 | 512 | 30k |
| `small` | 1 | 64 | 256 | 1 | 4 heads, window 64 | 64 | 1,024 | 219k |
| `base` | 2 | 96 | 160 | 1 | 4 heads, window 64 | 128 | 2,048 | 538k |
| `medium` | 2 | 128 | 256 | 1 | 4 heads, window 64 | 128 | 2,048 | 954k |
| `large` | 2 | 192 | 384 | 1 | 4 heads, window 64 | 128 | 2,048 | 1.95M |
| `pro` | 2 | 224 | 448 | 1 | 4 heads, window 64 | 128 | 2,048 | 2.57M |
| `xl` | 2 | 288 | 576 | 1 | 4 heads, window 64 | 128 | 2,048 (4k-8k advised) | 4.08M (5.9M at 8k) |

The order-search model uses `base` (538k) in the demo; the experiments found no gain above `large` (2M) and untuned learning
rates at `xl`. The model files named below are local artifacts (`models/` is gitignored).

Two command assistants came out of earlier work, both `base` with INT8 and decoding constrained to learned command words
(`-constrain`):

- **Git:** 63% exact match on `research/eval/git_tasks.json` against 46% for `small` (`models/git_base_sft_int8.json`,
  7.1 MB; `research/eval/sweeps/archive/RESULTS_constrained_decoding.md`).
- **Git + 73 Unix tools:** 65% on the git benchmark, 66% on full Unix commands and 94% on options it was trained on, but
  under 8% on commands it never saw (`models/assistant_base_sft_int8.json`; run it without `-constrain` for Unix;
  `research/eval/sweeps/archive/RESULTS_shell_task.md`).

TinyStories validation: `python research/tools/bench_tinystories.py` measures held-out bits per byte and generation speed, and
`research/tools/compare_story_samples.py` writes blinded story continuations to compare against HuggingFace checkpoints.

## Configs and sweeps

`train.Config` (JSON: defaults, then preset, then explicit keys) is shared by the CLI and the
sweep runner. Unknown keys are rejected, so a stale or misspelled key fails instead of silently
running the default. Live sweep specs go in `research/eval/sweeps/` (a test keeps them loadable); finished experiments with their logs and results are kept in `research/eval/sweeps/archive/`, whose specs may use config keys of since-removed features.

Checkpoints record the tokenizer format version (`tokenizer.FormatVersion`); loading a checkpoint
trained under other tokenizer rules is refused.

## Data

Nothing under `data/` or `models/` is in this repository. The corpus-building scripts in `research/tools` download
public sources (TinyStories, Simple English Wikipedia, Stack Exchange, tldr-pages, the git documentation) and
generate synthetic text with an LLM; each source keeps its own license, so check it before you redistribute a corpus
or a model trained on one. The model in `examples/order-search` is trained on small hand-written and synthetic data
about a made-up orders domain.

## License

Source-available under the [Elastic License 2.0](LICENSE): you can use, modify and redistribute it, including in commercial products, but you may not offer it to third parties as a hosted or managed service. The WebAssembly loader bundled in `packages/mignon-web` carries Go's BSD-3 license (see its `NOTICE`).
