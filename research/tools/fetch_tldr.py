#!/usr/bin/env python3
"""
fetch_tldr.py

Downloads tldr-pages (CC BY 4.0) and writes:
  data/sources/tldr.txt             pretraining blocks: page summary + one example + concrete command
  data/sources/tldr_git_pairs.json  SFT pairs from user-facing git pages ({"instruction", "output"});
                                    git's command-list.txt decides which subcommands are user-facing

tldr placeholders ({{path/to/file}}, {{branch_name}}, {{[-u|--include-untracked]}}) are filled with
realistic values from the build_git_sft entity generators, so the model sees real-looking commands.

Usage: tools/.venv/bin/python research/tools/fetch_tldr.py [--platforms common,linux] [--variants 2]
"""

import argparse
import io
import re
import urllib.request
import zipfile

import build_git_sft as ents
from techcorpus_common import DEFAULT_MAX_BLOCK_WORDS, dedupe, word_count, save_json, seeded_rng, source_path, write_blocks

TLDR_ZIP_URL = "https://codeload.github.com/tldr-pages/tldr/zip/refs/heads/main"
# Core git pages link to the official manual; third-party tools (git-extras, git-flow, ...) do not.
CORE_GIT_DOCS = "git-scm.com"
PAGE_RE = re.compile(r"^tldr-main/pages/([^/]+)/([^/]+)\.md$")
PLACEHOLDER_RE = re.compile(r"\{\{(.*?)\}\}")
OPTION_ALT_RE = re.compile(r"^\[(.+)\]$")
LONG_OPTION_PROB = 0.7
MAX_PLACEHOLDER_NUMBER = 10
GIT_COMMAND_LIST_URL = "https://raw.githubusercontent.com/git/git/master/command-list.txt"
PORCELAIN_CATEGORIES = {"mainporcelain", "ancillarymanipulators", "ancillaryinterrogators"}


def download_pages(platforms):
    print(f"Downloading {TLDR_ZIP_URL} ...")
    with urllib.request.urlopen(TLDR_ZIP_URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    pages = {}
    for name in archive.namelist():
        m = PAGE_RE.match(name)
        if m and m.group(1) in platforms:
            pages.setdefault(m.group(2), archive.read(name).decode("utf-8"))
    return pages


def parse_page(md):
    """Return (summary_sentences, [(example_description, command_template)], info_url)."""
    summary, examples, pending, info = [], [], None, ""
    for line in md.splitlines():
        line = line.strip()
        if line.startswith(">") and "More information" in line:
            info = line
        elif line.startswith(">"):
            summary.append(line.lstrip("> ").strip())
        elif line.startswith("- "):
            pending = line[2:].rstrip(":").strip()
        elif line.startswith("`") and line.endswith("`") and pending:
            examples.append((pending, line.strip("`")))
            pending = None
    return summary, examples, info


def placeholder_value(text, rng):
    """Concrete value for one tldr placeholder; returns (value, is_entity)."""
    alt = OPTION_ALT_RE.match(text)
    if alt or ("|" in text and " " not in text):
        # Alternatives like [-u|--include-untracked] or good|bad: prefer the last (long, readable) form.
        choices = (alt.group(1) if alt else text).split("|")
        return (choices[-1] if rng.random() < LONG_OPTION_PROB else choices[0]), False
    if text.startswith("-"):  # e.g. {{-number_of_commits}} -> -3
        value, is_entity = placeholder_value(text[1:], rng)
        return "-" + value, is_entity
    low = text.lower()
    # Order matters: "number_of_commits" is a number and "commit_message" a message, not a hash.
    for keys, gen in (
        (("number", "count"), lambda: str(rng.randint(1, MAX_PLACEHOLDER_NUMBER))),
        (("message",), ents.random_commit_message),
        (("branch",), ents.random_branch),
        (("commit", "hash", "sha", "revision"), ents.random_commit_hash),
        (("tag", "version"), ents.random_tag),
        (("url", "repository", "repo"), ents.random_url),
        (("email",), ents.random_email),
        (("remote",), lambda: rng.choice(ents.REMOTES)),
        (("name", "user", "author"), ents.random_username),
        (("path", "file", "directory", "dir"), ents.random_filepath),
    ):
        if any(k in low for k in keys):
            return gen(), True
    if low.isdigit():
        return text, False
    # Unknown placeholder: tldr names are readable, so keep the words instead of the braces.
    return text.replace("path/to/", "").replace("_", "-"), True


def fill(template, rng):
    """Fill every placeholder; returns (command, [entity values in order])."""
    entities = []

    def sub(m):
        value, is_entity = placeholder_value(m.group(1), rng)
        if is_entity:
            entities.append(value)
        return value

    return PLACEHOLDER_RE.sub(sub, template), entities


def download_porcelain():
    """Subcommands git itself lists as user-facing (main porcelain and ancillary commands)."""
    with urllib.request.urlopen(GIT_COMMAND_LIST_URL) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    names = set()
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#") and parts[1] in PORCELAIN_CATEGORIES:
            names.add(parts[0])
    return names


def valid_git_pair(page, command):
    """The command must run the page's own subcommand and have no unfilled template syntax."""
    if not page.startswith("git-"):  # standalone tools in git's command list (gitk, scalar, ...)
        return False
    subcommand = page[len("git-"):]
    return command.startswith(f"git {subcommand}") and not any(c in command for c in "{}")


def to_instruction(description, entities):
    """Lower-cased tldr description plus the concrete values, so the model learns to copy them."""
    text = description[0].lower() + description[1:] if description else description
    return f"{text}: {' '.join(entities)}" if entities else text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platforms", default="common,linux", help="tldr page folders to include")
    ap.add_argument("--variants", type=int, default=2, help="filled variants per git example (SFT pairs)")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_BLOCK_WORDS, help="max words per block")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = seeded_rng(args.seed)

    pages = download_pages(set(args.platforms.split(",")))
    porcelain = download_porcelain()
    blocks, git_pairs = [], []
    for name, md in sorted(pages.items()):
        summary, examples, info = parse_page(md)
        core_git = name in porcelain and CORE_GIT_DOCS in info
        for desc, template in examples:
            command, _ = fill(template, rng)
            # Keep summary sentences only while the block still fits the context budget.
            head = []
            for sentence in summary:
                if word_count(" ".join(head + [sentence, desc, command])) > args.max_words:
                    break
                head.append(sentence)
            blocks.append(f"{name}: {' '.join(head)}\n{desc}:\n{command}" if head else f"{desc}:\n{command}")
            if core_git:
                for _ in range(args.variants):
                    command, entities = fill(template, rng)
                    if not valid_git_pair(name, command):
                        continue
                    git_pairs.append({"instruction": to_instruction(desc, entities), "output": command})

    git_pairs = dedupe(git_pairs, key=lambda p: (p["instruction"], p["output"]))
    size = write_blocks(source_path("tldr.txt"), blocks)
    save_json(source_path("tldr_git_pairs.json"), git_pairs)
    print(f"{len(pages)} pages -> {len(blocks)} blocks ({size / 1e6:.1f} MB), {len(git_pairs)} git SFT pairs")


if __name__ == "__main__":
    main()
