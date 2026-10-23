#!/usr/bin/env python3
"""
generate_tinytech.py

Generates "TinyTech": short, simple-English paragraphs about computing and git, written by Claude
through the Message Batches API (or a local Ollama model with --backend ollama). This is the TinyStories recipe applied to technical English: a narrow,
consistent register is what a ~1M-parameter model can actually learn.

Output: data/sources/tinytech.txt (blank-line separated blocks; --out changes it)

Usage:
  tools/.venv/bin/python research/tools/generate_tinytech.py --requests 200 --dry-run   # inspect prompts, no API call
  tools/.venv/bin/python research/tools/generate_tinytech.py --requests 12000           # ~30 MB of text
Re-running the same command resumes the submitted batch (state in data/sources/tinytech_batch.json).
"""

import argparse

import claude_batch
import llm_backend
from techcorpus_common import (
    DEFAULT_MAX_BLOCK_WORDS, DEFAULT_MIN_BLOCK_WORDS, dedupe, is_mostly_ascii, normalize, seeded_rng,
    source_path, word_count, write_blocks,
)
from tinytech_seeds import FORMS, GENERAL_TOPICS, GIT_TOPICS, READERS, REQUIRED_WORDS

# Paragraphs asked for per request. A small local model loses count when asked for many, so it gets
# fewer per request (and more requests). It also overshoots a word limit, so it is asked for fewer words.
PARAGRAPHS_PER_REQUEST = {"claude": 8, "ollama": 2}
PROMPT_MAX_WORDS = {"claude": DEFAULT_MAX_BLOCK_WORDS, "ollama": 40}
REQUIRED_WORDS_PER_REQUEST = 3

PROMPT = """Write {n} separate short paragraphs for a training corpus that teaches simple technical English.

Topic: {topic}
Form: {form}
Reader: {reader}
Use each of these words at least once somewhere in the paragraphs: {words}

Rules:
- Use short sentences and common, everyday words. Only use a technical word when it is the topic or you explain it.
- Each paragraph has {min_words} to {max_words} words and makes sense on its own, without the others.
- Make the paragraphs different from each other: different situations, names, files and wording.
- Plain text only: no markdown, no bullet symbols, no headings.
- When you mention a command, write it inline exactly as it is typed, for example git status.
- Be correct. Do not invent commands or options."""

SCHEMA = claude_batch.json_list_schema("paragraphs", {"type": "string"})


def build_requests(n_requests, git_share, rng, backend, model, effort, per_request, max_words):
    requests = []
    for i in range(n_requests):
        topics = GIT_TOPICS if rng.random() < git_share else GENERAL_TOPICS
        prompt = PROMPT.format(
            n=per_request,
            topic=rng.choice(topics),
            form=rng.choice(FORMS),
            reader=rng.choice(READERS),
            words=", ".join(rng.sample(REQUIRED_WORDS, REQUIRED_WORDS_PER_REQUEST)),
            min_words=DEFAULT_MAX_BLOCK_WORDS // 2,
            max_words=max_words,
        )
        requests.append(backend.build_request(f"tinytech-{i}", prompt, SCHEMA, model=model, effort=effort))
    return requests


def clean_paragraphs(results):
    blocks = []
    for paragraphs in results.values():
        for p in paragraphs:
            p = normalize(p).replace("\n", " ")
            if DEFAULT_MIN_BLOCK_WORDS <= word_count(p) <= DEFAULT_MAX_BLOCK_WORDS and is_mostly_ascii(p):
                blocks.append(p)
    return dedupe(blocks, key=str.lower)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=int, default=1000, help="number of requests")
    ap.add_argument("--paragraphs-per-request", type=int, default=None,
                    help=f"paragraphs asked for in each request (default {PARAGRAPHS_PER_REQUEST})")
    ap.add_argument("--out", default=source_path("tinytech.txt"), help="output file")
    ap.add_argument("--git-share", type=float, default=0.35, help="share of requests on git topics")
    llm_backend.add_arguments(ap)
    ap.add_argument("--state", default=source_path("tinytech_batch.json"), help="batch id file for resuming")
    ap.add_argument("--dry-run", action="store_true", help="print a prompt and exit without calling the API")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    backend, model, effort = llm_backend.resolve(args)

    per_request = args.paragraphs_per_request or PARAGRAPHS_PER_REQUEST[args.backend]
    requests = build_requests(args.requests, args.git_share, seeded_rng(args.seed), backend, model, effort,
                              per_request, PROMPT_MAX_WORDS[args.backend])
    if args.dry_run:
        backend.dry_run(requests)
        return
    results = backend.run(requests, args.state, "paragraphs")
    returned = sum(len(p) for p in results.values())
    short = sum(len(p) < per_request for p in results.values())
    blocks = clean_paragraphs(results)
    print(f"{returned} paragraphs from {len(results)} requests ({short} returned fewer than the {per_request} asked "
          f"for); {len(blocks)} kept after the length, ASCII and duplicate filters")
    size = write_blocks(args.out, blocks)
    print(f"tinytech -> {len(blocks)} blocks ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
