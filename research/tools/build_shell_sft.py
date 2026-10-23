#!/usr/bin/env python3
"""
build_shell_sft.py

Builds a shell-command assistant task from tldr-pages (CC BY 4.0) as a second domain next to git:
  data/shell_sft.json          SFT pairs ({"instruction", "output"}) from tools the model may see
  data/shell_sft_heldout.json  pairs for the benchmark: "seen_intent" (new values for a trained command),
                               "unseen_intent" (new examples of trained tools) and "unseen_tool" pairs
  research/eval/shell_tasks.json        probes in the same format as research/eval/git_tasks.json

Git pages are skipped (the git task has its own data). tldr placeholders are filled with plausible
values, and the values are appended to the request ("extract an archive: backup.tar dest") so the
model can learn to copy them, as in the git data.

Usage: python3 research/tools/build_shell_sft.py --zip path/to/tldr.zip [--variants 3] [--per-category 60]
"""

import argparse
import collections
import io
import re
import zipfile

import build_git_sft as ents
import unix_glossary
from techcorpus_common import save_json, seeded_rng, stable_hash
from probes import copy_probe, option_probe, probe

PAGE_RE = re.compile(r"^tldr-main/pages/([^/]+)/([^/]+)\.md$")
PLACEHOLDER_RE = re.compile(r"\{\{(.*?)\}\}")
OPTION_ALT_RE = re.compile(r"^\[(.+)\]$")
MNEMONIC_RE = re.compile(r"\[(\w)\]")  # "[c]reate" -> "create"
PLATFORMS = ("common", "linux", "osx")
HELDOUT_TOOL_FRACTION = 0.08  # 0.08 of tools never appear in training
HELDOUT_PAIR_FRACTION = 0.08
HELDOUT_VARIANT_FRACTION = 0.15  # templates of seen tools that lose their last variant to the seen-intent probes
MAX_COMMAND_WORDS = 14
LONG_OPTION_PROB = 0.7

HOSTS = ["example.com", "server1", "db.internal", "10.0.0.5", "localhost", "web-01", "192.168.1.20"]
USERS = ["alice", "bob", "deploy", "root", "carol", "admin", "dev", "backup"]
WORDS = ["report", "data", "notes", "build", "logs", "config", "backup", "image", "archive", "session", "cache", "tmp"]
PATTERNS = ["error", "TODO", "main", "warning", "foo", "hello", "timeout", "failed", "user", "id"]
EXTS = ["txt", "log", "json", "csv", "md", "sh", "py", "yml", "conf", "tar.gz", "zip", "png", "pdf"]

TECH_ENGLISH = [
    ("To list the files in a folder, run", ["ls"]),
    ("To show the contents of a file, you can use the command", ["cat", "less"]),
    ("To search for text inside files, use", ["grep"]),
    ("To change to another directory, type", ["cd"]),
    ("To make a new folder, run", ["mkdir"]),
    ("To remove a file, run", ["rm"]),
    ("To copy a file, run", ["cp"]),
    ("To see how much disk space is free, run", ["df"]),
    ("To check which processes are running, run", ["ps", "top"]),
    ("To download a file from the web, you can use", ["curl", "wget"]),
    ("To make a script run, you must make it", ["executable"]),
    ("To connect to a remote server, use", ["ssh"]),
    ("To change who can read a file, use", ["chmod"]),
    ("The first line of a shell script is called the", ["shebang"]),
    ("Use the pipe symbol to send the output of one command to", ["another", "the"]),
    ("To print text to the screen, use", ["echo"]),
    ("To stop a running program, press", ["Ctrl"]),
    ("Run this command as the administrator with", ["sudo"]),
]


def read_pages(zip_path):
    archive = zipfile.ZipFile(zip_path)
    pages = {}
    for platform in PLATFORMS:
        for name in archive.namelist():
            m = PAGE_RE.match(name)
            if m and m.group(1) == platform:
                pages.setdefault(m.group(2), archive.read(name).decode("utf-8"))
    return pages


def parse_page(md):
    """(summary, [(description, command_template)])."""
    summary, examples, pending = [], [], None
    for line in md.splitlines():
        line = line.strip()
        if line.startswith(">") and "More information" not in line and "Deprecated" not in line:
            summary.append(line.lstrip("> ").strip().rstrip("."))
        elif line.startswith("- "):
            pending = line[2:].rstrip(":").strip()
        elif line.startswith("`") and line.endswith("`") and pending:
            examples.append((pending, line.strip("`")))
            pending = None
    return " ".join(summary[:1]), examples


def filename(rng):
    return f"{rng.choice(WORDS)}.{rng.choice(EXTS)}"


def placeholder_value(text, rng):
    """(value, copyable): copyable values are appended to the request."""
    alt = OPTION_ALT_RE.match(text)
    if alt or ("|" in text and " " not in text):
        # The same alternatives always give the same form, so a request maps to one command.
        choices = (alt.group(1) if alt else text).split("|")
        long_form = stable_hash(text, "option") % 1000 < LONG_OPTION_PROB * 1000
        return (choices[-1] if long_form else choices[0]), False
    low = text.lower()
    if text.startswith("-") and len(text) > 1:
        value, copy = placeholder_value(text[1:], rng)
        return "-" + value, copy
    if low.isdigit():
        return text, False
    if "..." in text:  # "path/to/file1 path/to/file2 ..." -> two concrete values
        base = text.split()[0].replace("...", "") or "file"
        first, _ = placeholder_value(base, rng)
        second, _ = placeholder_value(base, rng)
        return f"{first} {second}", True
    for keys, gen in (
        (("port",), lambda: str(rng.choice([22, 80, 443, 3000, 5432, 8080, 9000]))),
        (("seconds", "number", "count", "size", "lines", "depth", "level", "limit", "width"), lambda: str(rng.randint(1, 20))),
        (("pid",), lambda: str(rng.randint(100, 40000))),
        (("host", "domain", "ip", "address", "server"), lambda: rng.choice(HOSTS)),
        (("user", "owner", "group"), lambda: rng.choice(USERS)),
        (("email",), ents.random_email),
        (("url", "uri", "link"), ents.random_url),
        (("pattern", "regex", "string", "text", "keyword", "query", "search", "word"), lambda: rng.choice(PATTERNS)),
        (("directory", "folder", "dir"), lambda: rng.choice(["src", "docs", "build", "/var/log", "~/projects", "data", "/tmp/work"])),
        (("file", "path", "image", "archive", "script", "output", "input", "source", "target"), lambda: filename(rng)),
        (("name", "id", "tag", "label"), lambda: rng.choice(WORDS)),
    ):
        if any(k in low for k in keys):
            return gen(), True
    return text.replace("path/to/", "").replace("_", "-"), True


def fill(template, rng):
    values = []

    def sub(m):
        value, copy = placeholder_value(m.group(1), rng)
        if copy:
            values.append(value)
        return value

    return PLACEHOLDER_RE.sub(sub, template), values


def clean(description):
    text = MNEMONIC_RE.sub(r"\1", description)
    return text[0].lower() + text[1:] if text else text


def instruction_for(tool, description, values, rng):
    text = clean(description)
    style = rng.random()
    args = " ".join(values)
    if style < 0.45:
        out = text
    elif style < 0.75:
        out = f"with {tool}, {text}"
    else:
        out = f"how do I {text}?"
    return f"{out}: {args}" if args else out


def build_pairs(pages, variants, rng):
    """{tool: [(instruction, command, template)]} for non-git pages; variants share a template."""
    by_tool = collections.defaultdict(list)
    for tool, md in sorted(pages.items()):
        if tool.startswith("git"):
            continue
        _, examples = parse_page(md)
        for description, template in examples:
            for _ in range(variants):
                command, values = fill(template, rng)
                if "{" in command or "}" in command or len(command.split()) > MAX_COMMAND_WORDS:
                    continue
                if not command.split() or command.split()[0] != tool:
                    continue
                by_tool[tool].append((instruction_for(tool, description, values, rng), command, template))
    return by_tool


def build_tasks(heldout, per_category, rng):
    tasks = collections.defaultdict(list)
    rng.shuffle(heldout)
    for ex in heldout:
        inst, out = ex["instruction"], ex["output"]
        seen = {"seen_intent": "seen intent", "unseen_intent": "unseen intent", "unseen_tool": "unseen tool"}[ex["split"]]
        tasks[f"Shell full command ({seen})"].append(probe("", [out], inst, complete=True))
        opt = option_probe(out)
        if opt:
            tasks[f"Shell option ({seen})"].append(probe(opt[0], [opt[1]], inst))
        cp = copy_probe(inst, out)
        if cp:
            tasks[f"Shell argument copy ({seen})"].append(probe(cp[0], [cp[1]], inst))
    tasks["Technical English (shell)"] = [
        {"prompt": p, "expected": [" " + e for e in exp]} for p, exp in TECH_ENGLISH
    ]
    result = []
    for category, items in sorted(tasks.items()):
        seen_keys, unique = set(), []
        for t in items:
            key = (t["instruction"] if "instruction" in t else "", t["prompt"], tuple(t["expected"]))
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(t)
        unique = unique[:per_category]
        result.extend({"category": category, **t} for t in unique)
        print(f"  {category}: {len(unique)}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="tldr-pages main.zip")
    ap.add_argument("--variants", type=int, default=4, help="filled variants per example")
    ap.add_argument("--only", default="", help="comma-separated tool names; all other tools are skipped")
    ap.add_argument("--glossary-repeat", type=int, default=0,
                    help="copies of the unix_glossary.py training pairs mixed in (0 disables the glossary)")
    ap.add_argument("--per-category", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tasks", default="research/eval/shell_tasks.json")
    args = ap.parse_args()

    rng = seeded_rng(args.seed)
    pages = read_pages(args.zip)
    if args.only:
        pages = {t: md for t, md in pages.items() if t in args.only.split(",")}
    by_tool = build_pairs(pages, args.variants, rng)

    train, heldout = [], []
    heldout_tools = 0
    for tool, pairs in sorted(by_tool.items()):
        if not args.only and stable_hash(tool, "tool") % 1000 < HELDOUT_TOOL_FRACTION * 1000:
            heldout_tools += 1
            heldout += [{"instruction": i, "output": c, "split": "unseen_tool"} for i, c, _ in pairs]
            continue
        last_variant = {}
        for i, c, template in pairs:
            last_variant[template] = (i, c)
        for i, c, template in pairs:
            h = stable_hash(template, "pair") % 1000
            if h < HELDOUT_PAIR_FRACTION * 1000:
                heldout.append({"instruction": i, "output": c, "split": "unseen_intent"})
            elif h < (HELDOUT_PAIR_FRACTION + HELDOUT_VARIANT_FRACTION) * 1000 and last_variant[template] == (i, c):
                heldout.append({"instruction": i, "output": c, "split": "seen_intent"})
            else:
                train.append({"instruction": i, "output": c})

    if args.glossary_repeat > 0:
        g_train, g_seen, g_unseen = unix_glossary.build(seeded_rng(args.seed + 1))
        train += g_train * args.glossary_repeat
        heldout += [{**p, "split": "seen_intent"} for p in g_seen]
        heldout += [{**p, "split": "unseen_intent"} for p in g_unseen]
        print(f"glossary: {len(g_train)} train x {args.glossary_repeat}, {len(g_seen)} seen-intent and {len(g_unseen)} unseen-intent held out")
    save_json(f"{args.out_dir}/shell_sft.json", train)
    save_json(f"{args.out_dir}/shell_sft_heldout.json", heldout)
    print(f"{len(by_tool)} tools ({heldout_tools} held out): {len(train)} train pairs, {len(heldout)} held-out pairs")
    tasks = build_tasks(heldout, args.per_category, rng)
    save_json(args.tasks, tasks)
    print(f"{len(tasks)} probes -> {args.tasks}")


if __name__ == "__main__":
    main()
