#!/usr/bin/env python3
"""
fetch_commit_messages.py

Streams commit messages from bigcode/commitpackft (permissively licensed repos) and writes them as
blocks to data/sources/commits.txt. Commit messages are short imperative technical English
("Fix crash when config file is missing") - the register users write git requests in.

Only the message text is used; code contents are ignored.

Usage: tools/.venv/bin/python research/tools/fetch_commit_messages.py [--per-language 30000]
"""

import argparse
import re

from datasets import load_dataset

from techcorpus_common import (
    DEFAULT_MAX_BLOCK_WORDS, dedupe, is_mostly_ascii, source_path, take_paragraphs, write_blocks,
)

DATASET = "bigcode/commitpackft"
DEFAULT_LANGUAGES = "shell,python,javascript,go,c,java,markdown,yaml,dockerfile,rust,ruby,typescript"
MIN_MESSAGE_WORDS = 3

# Messages with no language in them: merges, version bumps, bare ticket ids.
BOILERPLATE_RE = re.compile(
    r"^(merge (pull request|branch|remote)|bump version|version bump|v?\d+(\.\d+)+$|wip$|update$|"
    r"[A-Z]+-\d+$|initial commit$)", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+")


def clean_message(row, max_words):
    message = URL_RE.sub("", row.get("message") or row.get("subject") or "")
    # Drop trailer lines (Signed-off-by, Co-authored-by, Change-Id, ...).
    lines = [ln for ln in message.splitlines() if not re.match(r"^[\w-]+:\s", ln.strip())]
    text = take_paragraphs("\n".join(lines), max_words)
    first = text.split("\n", 1)[0] if text else ""
    if len(first.split()) < MIN_MESSAGE_WORDS or BOILERPLATE_RE.match(first.strip()):
        return None
    return text if is_mostly_ascii(text) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default=DEFAULT_LANGUAGES, help="commitpackft subsets")
    ap.add_argument("--per-language", type=int, default=30000, help="messages kept per language")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_BLOCK_WORDS, help="max words per block")
    args = ap.parse_args()

    blocks = []
    for lang in args.languages.split(","):
        try:
            ds = load_dataset(DATASET, lang, split="train", streaming=True, trust_remote_code=True)
        except Exception as e:  # subset names differ between dataset revisions; skip missing ones
            print(f"  {lang}: skipped ({e.__class__.__name__})")
            continue
        kept = 0
        for row in ds:
            msg = clean_message(row, args.max_words)
            if msg:
                blocks.append(msg)
                kept += 1
                if kept >= args.per_language:
                    break
        print(f"  {lang}: {kept} messages")

    blocks = dedupe(blocks, key=str.lower)
    size = write_blocks(source_path("commits.txt"), blocks)
    print(f"commits -> {len(blocks)} blocks ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
