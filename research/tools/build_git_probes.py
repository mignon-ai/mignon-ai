#!/usr/bin/env python3
"""
build_git_probes.py

Builds probes for `mignon eval -benchmark -tasks research/eval/git_tasks.json` from the held-out git SFT pairs
(augment_git_sft.py) plus a fixed set of technical-English cloze prompts.

Each git probe holds the request as "instruction" and the start of the reply as "prompt"; the
benchmark wraps them in the chat template for fine-tuned models and joins them with a newline for
raw-text models. "expected" is the exact continuation, leading space included, because the
tokenizer gives each word its preceding space (" origin"). Categories:
  Git full command (seen|unseen intent)  ""             -> "git stash -u"  (must end the reply)
  Git subcommand (seen|unseen intent)    "git"          -> " stash"
  Git option (seen|unseen intent)        "git stash"    -> " -u"
  Git argument copy                      "git switch"   -> " dev", a value that is in the request
  Technical English                      "To save your work in git, you make a" -> " commit"

Usage: tools/.venv/bin/python research/tools/build_git_probes.py [--per-category 60]
"""

import argparse
import collections
import json

from techcorpus_common import load_json, save_json, seeded_rng

HELDOUT = "data/git_sft_v2_heldout.json"
OUTPUT = "research/eval/git_tasks.json"

TECH_ENGLISH = [
    ("To save a snapshot of your work in git, you make a", ["commit"]),
    ("Before you commit a file, you have to add it to the staging", ["area"]),
    ("A copy of a repository on your own computer is called a local", ["copy", "repository", "clone"]),
    ("To get the latest changes from the server, run git", ["pull", "fetch"]),
    ("To send your commits to the server, run git", ["push"]),
    ("To see which files changed, run git", ["status", "diff"]),
    ("Two people changed the same line, so git reported a merge", ["conflict"]),
    ("Each commit has a short message that describes the", ["change", "changes", "commit"]),
    ("A branch lets you work on a new feature without changing the main", ["branch"]),
    ("If the program crashes, read the error", ["message"]),
    ("Open a terminal and type the", ["command"]),
    ("The file is saved in the project", ["folder", "directory"]),
    ("Run the tests before you", ["commit", "push", "merge"]),
    ("Files listed in .gitignore are not", ["tracked", "added", "committed"]),
    ("To download a repository for the first time, run git", ["clone"]),
    ("You can undo the last commit with git", ["reset", "revert"]),
    ("The server rejected the push because the remote has new", ["commits", "changes"]),
    ("Your name and email are stored in the git", ["config", "configuration"]),
]


def option_probe(command):
    """(prefix, " option") for the first option after the subcommand, if any."""
    parts = command.split()
    for i, tok in enumerate(parts[2:], start=2):
        if tok.startswith("-"):
            return " ".join(parts[:i]), " " + tok.split("=", 1)[0]
        break
    return None


def copy_probe(instruction, command):
    """(prefix, " argument") for the first argument that is copied verbatim from the request."""
    parts = command.split()
    for i, tok in enumerate(parts[2:], start=2):
        value = tok.strip("'\"")
        if not tok.startswith("-") and len(value) > 1 and value in instruction:
            # A quoted argument: the opening quote and its space belong to the prompt.
            if tok[0] in "'\"":
                return " ".join(parts[:i]) + " " + tok[0], value
            return " ".join(parts[:i]), " " + value
    return None


def probe(prompt, expected, instruction="", complete=False):
    """A task dict without category (build_tasks adds it)."""
    task = {}
    if instruction:
        task["instruction"] = instruction
    task.update({"prompt": prompt, "expected": list(expected)})
    if complete:
        task["complete"] = True
    return task


def build_tasks(heldout, per_category, rng):
    tasks = collections.defaultdict(list)
    rng.shuffle(heldout)
    for ex in heldout:
        inst, out = ex["instruction"], ex["output"].split("&&")[0].strip()
        parts = out.split()
        if len(parts) < 2 or parts[0] != "git":
            continue
        seen = "seen" if ex.get("split") == "seen_intent" else "unseen"
        tasks[f"Git full command ({seen} intent)"].append(probe("", [out], inst, complete=True))
        tasks[f"Git subcommand ({seen} intent)"].append(probe("git", [" " + parts[1]], inst))
        opt = option_probe(out)
        if opt:
            tasks[f"Git option ({seen} intent)"].append(probe(opt[0], [opt[1]], inst))
        cp = copy_probe(inst, out)
        if cp:
            tasks["Git argument copy"].append(probe(cp[0], [cp[1]], inst))
    tasks["Technical English"] = [probe(p, [" " + e for e in exp]) for p, exp in TECH_ENGLISH]

    out = []
    for category, items in sorted(tasks.items()):
        unique = list({json.dumps(t, sort_keys=True): t for t in items}.values())[:per_category]
        out.extend({"category": category, **t} for t in unique)
        print(f"  {category}: {len(unique)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-category", type=int, default=60, help="max probes per category")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tasks = build_tasks(load_json(HELDOUT), args.per_category, seeded_rng(args.seed))
    save_json(OUTPUT, tasks)
    print(f"{len(tasks)} probes -> {OUTPUT}")


if __name__ == "__main__":
    main()
