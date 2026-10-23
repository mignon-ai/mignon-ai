#!/usr/bin/env python3
"""
fetch_git_docs.py

Downloads git's own documentation (GPLv2) and, optionally, the Pro Git book (CC BY-NC-SA 3.0,
non-commercial only) and converts the AsciiDoc to plain-text blocks:
  data/sources/git_docs.txt
  data/sources/progit.txt   (only with --progit)

The man pages are dense reference text. Their DESCRIPTION/EXAMPLES prose is what teaches the model
how git is talked about, so option tables and synopsis blocks are dropped by default.

Usage: tools/.venv/bin/python research/tools/fetch_git_docs.py [--progit] [--keep-options]
"""

import argparse
import io
import re
import tarfile
import urllib.request

from techcorpus_common import DEFAULT_MAX_BLOCK_WORDS, source_path, text_to_blocks, unwrap_lines, write_blocks

GIT_TARBALL_URL = "https://codeload.github.com/git/git/tar.gz/refs/heads/master"
PROGIT_TARBALL_URL = "https://codeload.github.com/progit/progit2/tar.gz/refs/heads/main"
DOC_EXTENSIONS = (".adoc", ".txt", ".asc")

# Man-page sections whose content is mostly option lists or syntax, not prose.
SKIP_SECTIONS = {"SYNOPSIS", "OPTIONS", "CONFIGURATION", "SEE ALSO", "GIT", "AUTHOR", "DOCUMENTATION"}

_SECTION_RE = re.compile(r"^(?:==+\s+)?([A-Z][A-Z \-]+)$")
_DELIMITER_RE = re.compile(r"^(?:-{4,}|\.{4,}|\+{4,}|_{4,}|\|===)$")
_UNDERLINE_RE = re.compile(r"^[-=~^+.*_]{3,}$")
_SKIP_LINE_RE = re.compile(r"^(?:include::|ifdef::|ifndef::|endif::|ifeval::|:[\w-]+:|\[\[|\[[\w ,=\"'.#-]*\]$|//|image::|toc::)")
_INLINE_SUBS = (
    (re.compile(r"(?:linkgit|link|xref|<<)[:]?([\w.\-/#]+)(?:\[[^\]]*\]|>>)"), r"\1"),
    (re.compile(r"\{litdd\}"), "--"),
    (re.compile(r"\{[\w-]+\}"), ""),
    (re.compile(r"(?<!\w)([`*_+]{1,2})([^`*_+\n]+?)\1(?!\w)"), r"\2"),
    (re.compile(r"\(\(+([^)]+)\)+\)"), r"\1"),  # index terms ((term))
)


def download_tar(url):
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        return tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")


def asciidoc_to_text(src, keep_options):
    """Strip AsciiDoc markup, keeping paragraphs separated by blank lines."""
    out, section, in_listing, skip_underline = [], "", False, False
    lines = src.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if skip_underline:  # "----" under a section title is not a listing delimiter
            skip_underline = False
            continue
        if _DELIMITER_RE.match(stripped):
            in_listing = not in_listing
            out.append("")
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        title = _SECTION_RE.match(stripped)
        if title and (stripped.startswith("=") or _UNDERLINE_RE.match(nxt)):
            section = title.group(1).strip()
            skip_underline = not stripped.startswith("=")
            out.append("")
            continue
        if _UNDERLINE_RE.match(stripped) or _SKIP_LINE_RE.match(stripped):
            out.append("")
            continue
        if not keep_options and section in SKIP_SECTIONS:
            continue
        if stripped.startswith("$ "):  # shell prompt in front of an example command
            stripped = stripped[2:]
        if stripped.startswith("="):  # Pro Git chapter/section titles in mixed case
            stripped = stripped.lstrip("= ").strip()
        if stripped.endswith("::"):  # definition-list term ("--force::"), keep as its own line
            stripped = stripped.rstrip(":") + ":"
        # Indented lines are AsciiDoc literal blocks: treat them like listings.
        if in_listing or (line[:1] in (" ", "\t") and stripped):
            # Keep commands from example listings; drop program output and long dumps.
            if stripped.startswith("$ "):
                out.append(stripped[2:])
            elif stripped.startswith("git "):
                out.append(stripped)
            continue
        for pattern, repl in _INLINE_SUBS:
            stripped = pattern.sub(repl, stripped)
        out.append(stripped.lstrip("+").strip())
    return "\n".join(out)


def convert(tar, path_filter, keep_options, max_words):
    blocks = []
    for member in tar.getmembers():
        if member.isfile() and member.name.endswith(DOC_EXTENSIONS) and path_filter(member.name):
            src = tar.extractfile(member).read().decode("utf-8", errors="replace")
            text = "\n\n".join(unwrap_lines(p) for p in asciidoc_to_text(src, keep_options).split("\n\n"))
            blocks.extend(text_to_blocks(text, max_words))
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--progit", action="store_true", help="also fetch Pro Git (CC BY-NC-SA, non-commercial)")
    ap.add_argument("--keep-options", action="store_true", help="keep OPTIONS/SYNOPSIS/CONFIGURATION sections")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_BLOCK_WORDS, help="max words per block")
    args = ap.parse_args()

    git_tar = download_tar(GIT_TARBALL_URL)
    in_docs = lambda n: "/Documentation/" in n and "/RelNotes/" not in n and "/technical/" not in n
    blocks = convert(git_tar, in_docs, args.keep_options, args.max_words)
    size = write_blocks(source_path("git_docs.txt"), blocks)
    print(f"git docs -> {len(blocks)} blocks ({size / 1e6:.1f} MB)")

    if args.progit:
        book_tar = download_tar(PROGIT_TARBALL_URL)
        blocks = convert(book_tar, lambda n: "/book/" in n, True, args.max_words)
        size = write_blocks(source_path("progit.txt"), blocks)
        print(f"Pro Git -> {len(blocks)} blocks ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
