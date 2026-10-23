#!/usr/bin/env python3
"""
fetch_web_tech.py

Streams RefinedWeb (ODC-BY) and keeps only pages whose vocabulary is technical enough, then splits
them into context-sized blocks: data/sources/web_tech.txt

This keeps a slice of general web English in the mix without spending model capacity on
fashion blogs and recipes.

Usage: tools/.venv/bin/python research/tools/fetch_web_tech.py [--target-mb 15] [--min-density 0.03]
"""

import argparse
import re

from datasets import load_dataset

from techcorpus_common import DEFAULT_MAX_BLOCK_WORDS, is_mostly_ascii, looks_like_prose, source_path, text_to_blocks, write_blocks

DATASET = "tiiuae/falcon-refinedweb"
SHUFFLE_BUFFER = 10000
MIN_PAGE_WORDS = 80
BYTES_PER_MB = 1_000_000

TECH_TERMS = (
    "software code program programming developer computer server terminal command shell linux unix "
    "windows macos file files folder directory repository repo git github version commit branch merge "
    "install installed configure configuration config script scripts api database network browser "
    "update upgrade bug error debug compile build deploy package library framework python java "
    "javascript open-source source cloud backup password user users admin login data system systems "
    "application app apps website web html css binary download upload storage disk memory cpu"
).split()
WORD_RE = re.compile(r"[a-z][a-z\-]+")


def tech_density(text, terms):
    words = WORD_RE.findall(text.lower())
    if len(words) < MIN_PAGE_WORDS:
        return 0.0
    return sum(1 for w in words if w in terms) / len(words)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-mb", type=float, default=15.0, help="stop after this much kept text")
    ap.add_argument("--min-density", type=float, default=0.03, help="min share of words that are tech terms")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_BLOCK_WORDS, help="max words per block")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    terms = frozenset(TECH_TERMS)
    ds = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=args.seed, buffer_size=SHUFFLE_BUFFER)
    blocks, kept_bytes, scanned, pages = [], 0, 0, 0
    target = args.target_mb * BYTES_PER_MB
    for row in ds:
        scanned += 1
        text = row["content"]
        if tech_density(text, terms) < args.min_density or not is_mostly_ascii(text):
            continue
        page_blocks = [b for b in text_to_blocks(text, args.max_words) if looks_like_prose(b)]
        blocks.extend(page_blocks)
        kept_bytes += sum(len(b) for b in page_blocks)
        pages += 1
        if pages % 1000 == 0:
            print(f"  scanned {scanned}, kept {pages} pages ({kept_bytes / BYTES_PER_MB:.1f} MB)")
        if kept_bytes >= target:
            break

    size = write_blocks(source_path("web_tech.txt"), blocks)
    print(f"web_tech -> {pages}/{scanned} pages, {len(blocks)} blocks ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
