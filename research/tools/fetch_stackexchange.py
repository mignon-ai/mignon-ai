#!/usr/bin/env python3
"""
fetch_stackexchange.py

Streams Stack Exchange Q&A (CC BY-SA, via HuggingFaceH4/stack-exchange-preferences) from technical
sites and writes short "question / best answer" blocks:
  data/sources/stackexchange.txt        all selected sites
  data/sources/stackexchange_git.txt    only threads that mention git (for the git-focused stage)

Only the leading paragraphs of each question and answer are kept, so a thread fits one context window.

Usage: tools/.venv/bin/python research/tools/fetch_stackexchange.py [--max-per-site 20000] [--min-score 2]
"""

import argparse
import re

from datasets import load_dataset

from techcorpus_common import (
    DEFAULT_MAX_BLOCK_WORDS, html_to_text, is_mostly_ascii, source_path, take_paragraphs, write_blocks,
)

DATASET = "HuggingFaceH4/stack-exchange-preferences"

# site -> whether a thread must mention git to be kept. Stack Overflow is too broad to take whole.
DEFAULT_SITES = {
    "Stackoverflow.com": True,
    "unix.stackexchange.com": False,
    "askubuntu.com": False,
    "superuser.com": False,
    "serverfault.com": False,
    "softwareengineering.stackexchange.com": False,
    "devops.stackexchange.com": False,
}

GIT_RE = re.compile(r"\bgit\b|\bgithub\b|\bgitlab\b", re.IGNORECASE)
# Share of the block's word budget given to the question; the answer gets the rest.
QUESTION_BUDGET_SHARE = 0.4


def best_answer(answers, min_score):
    """Accepted answer if it qualifies, otherwise the highest-scored one."""
    good = [a for a in answers if a.get("pm_score", 0) >= min_score]
    if not good:
        return None
    return max(good, key=lambda a: (a.get("selected", False), a.get("pm_score", 0)))


def thread_block(row, min_score, max_words):
    answer = best_answer(row.get("answers") or [], min_score)
    if answer is None:
        return None
    q_budget = int(max_words * QUESTION_BUDGET_SHARE)
    question = take_paragraphs(html_to_text(row["question"]), q_budget)
    reply = take_paragraphs(html_to_text(answer["text"]), max_words - len(question.split()))
    if not question or not reply:
        return None
    block = f"{question}\n{reply}"
    return block if is_mostly_ascii(block) else None


def fetch_site(site, git_only, args):
    ds = load_dataset(DATASET, data_dir=f"data/{site}", split="train", streaming=True)
    blocks, scanned = [], 0
    for row in ds:
        scanned += 1
        if git_only and not GIT_RE.search(row["question"]):
            continue
        block = thread_block(row, args.min_score, args.max_words)
        if block:
            blocks.append(block)
            if len(blocks) >= args.max_per_site:
                break
        if args.max_scan and scanned >= args.max_scan:
            break
    print(f"  {site}: scanned {scanned}, kept {len(blocks)}")
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES), help="comma-separated site folders")
    ap.add_argument("--max-per-site", type=int, default=20000, help="blocks kept per site")
    ap.add_argument("--max-scan", type=int, default=0, help="rows scanned per site (0 = no limit)")
    ap.add_argument("--min-score", type=int, default=2, help="minimum answer score")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_BLOCK_WORDS, help="max words per block")
    args = ap.parse_args()

    all_blocks, git_blocks = [], []
    for site in args.sites.split(","):
        blocks = fetch_site(site, DEFAULT_SITES.get(site, False), args)
        all_blocks.extend(blocks)
        git_blocks.extend(b for b in blocks if GIT_RE.search(b))

    size = write_blocks(source_path("stackexchange.txt"), all_blocks)
    git_size = write_blocks(source_path("stackexchange_git.txt"), git_blocks)
    print(f"stackexchange -> {len(all_blocks)} blocks ({size / 1e6:.1f} MB), "
          f"git subset {len(git_blocks)} ({git_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
