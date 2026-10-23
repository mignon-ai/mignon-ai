# Research

Work that produced the model design and the tuned presets, kept apart from the engine. Nothing in `pkg/` or
`cmd/` depends on it.

| Path | What |
|---|---|
| `tools/` | Corpus building and data generation for the git and Unix command assistants, the tech-text curriculum and the TinyStories benchmarks. Python; `build_tech_corpus.py` documents the order of the steps |
| `FINDINGS.md` | What limits a small model on a narrow task: the order-model experiments, results and caveats |
| `eval/order/` | Order-search test sets `hw5`-`hw8` (see FINDINGS.md for what each is worth) |
| `eval/git_tasks.json`, `eval/unix_tasks.json` | Benchmark probes for `mignon eval -benchmark -tasks ...` |
| `eval/sweeps/` | Sweep specs (`mignon sweep ...`), and in `archive/` the finished experiments with their results |

## Running the scripts

They are written to run from the repository root and import shared helpers from `tools/` (`techcorpus_common`,
`llm_backend`, `claude_batch`), so put `tools/` on the path:

```bash
PYTHONPATH=tools python3 research/tools/build_shell_sft.py --zip path/to/tldr.zip
(cd research/tools && PYTHONPATH=../../tools python3 -m unittest discover)   # their tests
```
