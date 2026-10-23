#!/usr/bin/env python3
"""
generate_paraphrases.py

Adds phrasing variety to vetted (instruction, command) pairs. A language model, by default a local
Ollama model, only rewrites the request; the command is copied from the source pair, so the model
cannot introduce a wrong flag. Every paraphrase must keep the values of the original request (file
names, numbers, patterns) and must pass the checks in valid_paraphrase.

Input:  a pairs JSON ([{"instruction", "output"}]), for example the shell SFT data.
Output: a pairs JSON with the accepted paraphrases ({"instruction", "output", "source"}).

Usage:
  python3 tools/generate_paraphrases.py --pairs data/shell_sft.json --out data/shell_paraphrases.json \
      --backend ollama --per-tool 20 --state data/shell_paraphrases_state.json
Re-running with the same arguments resumes: finished requests are read from the state file.
"""

import argparse
import collections
import re

import llm_backend
from techcorpus_common import dedupe, is_mostly_ascii, load_json, save_json, seeded_rng, word_count

PARAPHRASES_PER_REQUEST = 3
MIN_WORDS, MAX_WORDS = 3, 40
# A paraphrase much longer than its source has usually added details the command does not do.
MAX_LENGTH_RATIO, MIN_LENGTH_LIMIT = 2.0, 15
REFUSAL_RE = re.compile(r"\b(here are|i cannot|i can't|as an ai|paraphrase)\b", re.IGNORECASE)

PROMPT = """A user asked a command-line assistant:
"{instruction}"
The assistant's correct answer is the command: {command}

Write {k} different ways a user could ask for exactly the same thing.

Rules:
- Keep every file name, directory, number and other value exactly as in the original request.
- Do not change the meaning. Do not mention the command itself or put code in the request.
- Vary the wording, length and tone: a short direct request, a polite question, a casual phrase.
- Plain text only."""


def is_name_or_number(value):
    return any(c.isdigit() for c in value) or value[0].isupper()


def value_tokens(instruction, command, only_names=False):
    """Arguments of the command that the request mentions: they must survive paraphrasing.
    With only_names, just numbers and capitalised names: words like unpaid or find may be reworded."""
    values = []
    for tok in command.split()[1:]:
        value = tok.strip("'\"")
        if not value.startswith("-") and len(value) > 1 and value in instruction:
            if not only_names or is_name_or_number(value):
                values.append(value)
    return values


def valid_paraphrase(paraphrase, instruction, command, only_names=False):
    text = paraphrase.strip()
    if not (MIN_WORDS <= word_count(text) <= MAX_WORDS) or not is_mostly_ascii(text):
        return False
    if word_count(text) > max(MIN_LENGTH_LIMIT, MAX_LENGTH_RATIO * word_count(instruction)):
        return False
    if text.lower() == instruction.strip().lower() or REFUSAL_RE.search(text) or "`" in text:
        return False
    if command in text:  # the request must not contain the answer
        return False
    return all(v in text for v in value_tokens(instruction, command, only_names))


def sample_pairs(pairs, per_tool, rng):
    """Distinct pairs spread evenly over tools (the first word of the command)."""
    by_tool = collections.defaultdict(list)
    for p in dedupe(pairs, key=lambda p: (p["instruction"], p["output"])):
        by_tool[p["output"].split()[0]].append(p)
    chosen = []
    for tool in sorted(by_tool):
        group = by_tool[tool]
        rng.shuffle(group)
        chosen.extend(group[:per_tool])
    rng.shuffle(chosen)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="source pairs JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", required=True, help="results file for resuming")
    ap.add_argument("--per-tool", type=int, default=20, help="source pairs per tool")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N sampled pairs (0 = all)")
    ap.add_argument("--k", type=int, default=PARAPHRASES_PER_REQUEST, help="paraphrases asked per pair")
    ap.add_argument("--only-names", action="store_true",
                    help="require only numbers and capitalised names to survive (other words may be reworded)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    llm_backend.add_arguments(ap)
    ap.set_defaults(backend="ollama")
    args = ap.parse_args()
    backend, model, effort = llm_backend.resolve(args)

    sources = sample_pairs(load_json(args.pairs), args.per_tool, seeded_rng(args.seed))
    if args.limit:
        sources = sources[:args.limit]
    schema = backend.json_list_schema("requests", {"type": "string"})
    requests = [
        backend.build_request(f"para-{i}", PROMPT.format(instruction=p["instruction"], command=p["output"], k=args.k),
                              schema, model=model, effort=effort)
        for i, p in enumerate(sources)
    ]
    if args.dry_run:
        backend.dry_run(requests)
        return

    results = backend.run(requests, args.state, "requests")
    accepted, asked, rejected = [], 0, 0
    for cid, texts in results.items():
        src = sources[int(cid.rsplit("-", 1)[1])]
        for text in texts[:args.k]:
            asked += 1
            if valid_paraphrase(text, src["instruction"], src["output"], args.only_names):
                accepted.append({"instruction": text.strip(), "output": src["output"], "source": src["instruction"]})
            else:
                rejected += 1
    accepted = dedupe(accepted, key=lambda p: (p["instruction"].lower(), p["output"]))
    save_json(args.out, accepted)
    print(f"{len(sources)} source pairs, {len(results)} answered: {asked} paraphrases, {rejected} rejected, "
          f"{len(accepted)} kept -> {args.out}")


if __name__ == "__main__":
    main()
