"""
git_intents.py

Groups git SFT examples by *intent*: the command skeleton with entity values removed.
"push origin feature/login" and "push upstream main" share the intent "git push <arg> <arg>".

Held-out splits are made per intent, so the evaluation measures whether the model learned a
command it has never been trained on, rather than a seen template with a new branch name.
"""

import re

from techcorpus_common import HASH_RANGE, stable_hash

ARG = "<arg>"
_GIT_CMD_RE = re.compile(r"\bgit [^`'\"\n]+")


def command_skeleton(command):
    """'git push -u origin main' -> 'git push -u <arg> <arg>' (first command of a && chain)."""
    parts = command.strip().split("&&")[0].split()
    if len(parts) < 2 or parts[0] != "git":
        return ""
    skeleton = ["git", parts[1]]
    for tok in parts[2:]:
        if tok.startswith("-"):
            skeleton.append(tok.split("=", 1)[0])
        elif skeleton[-1] != ARG:
            skeleton.append(ARG)
    return " ".join(skeleton)


def chain_skeleton(command):
    """Skeleton of every command in a && chain, joined with ' && '."""
    return " && ".join(filter(None, (command_skeleton(c) for c in command.split("&&"))))


def intent_key(example):
    """Intent of an SFT example: the skeleton of the command it answers with, or asks about."""
    out = example["output"].strip()
    if out.startswith("git "):
        return chain_skeleton(out)
    mentioned = _GIT_CMD_RE.search(example["instruction"])
    if not mentioned:
        return "text: " + out[:40].lower()
    # In free text the command's end is unknown ("what does git stash -u do?"), so keep flags only.
    flags_only = [t for t in command_skeleton(mentioned.group(0)).split() if t != ARG]
    return "about: " + " ".join(flags_only)


def subcommand(command):
    parts = command.split()
    return parts[1] if len(parts) > 1 and parts[0] == "git" else ""


def is_heldout(key, fraction, salt="git-sft-heldout"):
    """Deterministic per-intent assignment, stable across runs and data additions."""
    return stable_hash(key, salt) / HASH_RANGE < fraction
