#!/usr/bin/env python3
"""
fetch_simple_wiki.py

Reads the Simple English Wikipedia dump (CC BY-SA, https://dumps.wikimedia.org/simplewiki/) and writes its prose
as context-sized blocks: data/sources/simplewiki.txt

Simple English Wikipedia is written with a small vocabulary and short sentences, so it is general
English a ~1M-parameter model can learn. It feeds the optional "english" stage of
build_tech_corpus.py (an English-first curriculum before the technical corpus).

Standard library only. Download the dump once, then:
  curl -LO https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
  tools/.venv/bin/python research/tools/fetch_simple_wiki.py --dump simplewiki-latest-pages-articles.xml.bz2 [--target-mb 150]
"""

import argparse
import bz2
import html
import random
import re
import xml.etree.ElementTree as ET

from techcorpus_common import (
    DEFAULT_MAX_BLOCK_WORDS, dedupe, is_mostly_ascii, looks_like_prose, source_path, text_to_blocks,
    write_blocks,
)

# Stubs and lists ("X is a commune in Y.") teach templates rather than language.
MIN_ARTICLE_WORDS = 120
BYTES_PER_MB = 1_000_000
PROGRESS_EVERY = 5000
# Sections that hold lists and references, not prose.
TAIL_SECTIONS = ("references", "other websites", "related pages", "notes", "sources", "external links", "further reading")

_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_FILE_RE = re.compile(r"\[\[(?:File|Image|Category|[a-z]{2,3}):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]", re.IGNORECASE)
_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_EXTLINK_RE = re.compile(r"\[https?://\S+(?:\s+([^\]]*))?\]")
_QUOTES_RE = re.compile(r"'{2,}")
_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$")


def strip_templates(text):
    """Removes {{...}} templates, innermost first, which also handles nesting."""
    while True:
        out = _TEMPLATE_RE.sub("", text)
        if out == text:
            return out
        text = out


def clean_wikitext(text):
    """Prose of an article: no markup, no tables, lists, headings or trailing reference sections."""
    text = _COMMENT_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    text = strip_templates(text)
    text = _FILE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _EXTLINK_RE.sub(lambda m: m.group(1) or "", text)
    text = _QUOTES_RE.sub("", text)
    text = _TAG_RE.sub("", html.unescape(html.unescape(text)))  # the dump escapes some entities twice
    paragraphs = []
    for line in text.split("\n"):
        line = line.strip()
        heading = _HEADING_RE.match(line)
        if heading:
            if heading.group(1).lower() in TAIL_SECTIONS:
                break
            continue
        # list items, indented lines and leftover markup are not prose
        if (not line or line[0] in "*#:;|!{}[]" or "[[" in line or "{{" in line or "\\" in line
                or line.lower().startswith(("file:", "image:", "category:"))):
            continue
        paragraphs.append(line)
    return "\n\n".join(paragraphs)


def articles(dump_path):
    """Yields (title, wikitext) of the main-namespace, non-redirect pages in the dump."""
    with bz2.open(dump_path, "rb") as f:
        for _, elem in ET.iterparse(f):
            if not elem.tag.endswith("}page"):
                continue
            ns = elem.find("{*}ns")
            redirect = elem.find("{*}redirect")
            text = elem.find("{*}revision/{*}text")
            title = elem.findtext("{*}title")
            if ns is not None and ns.text == "0" and redirect is None and text is not None and text.text:
                yield title, text.text
            elem.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="simplewiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--target-mb", type=float, default=150.0, help="stop after this much kept text")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_BLOCK_WORDS, help="max words per block")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # The dump is ordered by page id, so shuffle the kept articles before cutting at the target size.
    kept, scanned = [], 0
    for title, wikitext in articles(args.dump):
        scanned += 1
        text = clean_wikitext(wikitext)
        if len(text.split()) >= MIN_ARTICLE_WORDS and is_mostly_ascii(text):
            kept.append(text)
        if scanned % PROGRESS_EVERY == 0:
            print(f"  scanned {scanned}, kept {len(kept)} articles")
    random.Random(args.seed).shuffle(kept)

    blocks, kept_bytes, used = [], 0, 0
    target = args.target_mb * BYTES_PER_MB
    for text in kept:
        article_blocks = [b for b in text_to_blocks(text, args.max_words) if looks_like_prose(b)]
        blocks.extend(article_blocks)
        kept_bytes += sum(len(b) for b in article_blocks)
        used += 1
        if kept_bytes >= target:
            break

    blocks = dedupe(blocks, key=str.lower)
    size = write_blocks(source_path("simplewiki.txt"), blocks)
    print(f"simplewiki -> {used}/{scanned} articles, {len(blocks)} blocks ({size / BYTES_PER_MB:.1f} MB)")


if __name__ == "__main__":
    main()
