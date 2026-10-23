"""
git_sft_render.py

Renders git SFT (prompt, command) pairs from templates so that a command never contains an
argument the prompt does not name, and gives fixed-phrase intents more than a handful of wordings.

Prompt and command templates use named fields ("push {branch} to {remote}" ->
"git push {remote} {branch}"). A field the chosen prompt leaves out is treated the way git treats a
missing positional argument: dropped when no later field is given, otherwise filled with git's own
default (OMITTED_DEFAULTS). "push {branch}" thus becomes "git push origin {branch}", and "how do I
push all tags?" becomes "git push --tags" instead of naming a random remote.
"""

import random
import string

# Git's default for a positional argument that must be filled because a later one is given.
OMITTED_DEFAULTS = {"remote": "origin"}

# Wrappers that turn an imperative request into other natural phrasings of the same request.
PHRASINGS = [
    "{}",
    "how do I {}?",
    "how to {}",
    "how can I {}?",
    "I want to {}",
    "I need to {}",
    "please {}",
    "can you {}?",
]

# Requests shorter than this many words ("stash", "diff staged") read badly inside a wrapper.
MIN_REPHRASE_WORDS = 3

# Prompts that are already phrased (questions, polite requests) or are a literal git command.
_PHRASED_STARTS = ("how ", "what ", "why ", "which ", "when ", "where ", "can ", "please ", "git ")


def template_fields(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def ground_command(command, named, values):
    """Fill command with the fields in named; drop or default the others (see module doc)."""
    tokens = command.split()
    fields = [template_fields(t) for t in tokens]
    last_named = max((i for i, f in enumerate(fields) if f & named), default=-1)
    out = []
    for i, (tok, f) in enumerate(zip(tokens, fields)):
        missing = f - named
        if not missing:
            out.append(tok.format(**values))
        elif i < last_named:
            if not missing <= OMITTED_DEFAULTS.keys():
                raise ValueError(f"no default for {sorted(missing)} needed by {command!r}")
            out.append(tok.format(**{**values, **OMITTED_DEFAULTS}))
    return " ".join(out)


def pick(prompts, command, **values):
    """Choose a prompt template and render it with the command grounded in what it names."""
    prompt = random.choice(prompts)
    return prompt.format(**values), ground_command(command, template_fields(prompt), values)


def is_imperative(prompt):
    lowered = prompt.lower()
    return not lowered.startswith(_PHRASED_STARTS) and not lowered.endswith("?")


def rephrase(prompt):
    """Wrap an imperative prompt in a random phrasing; questions and short requests stay as they are."""
    if not is_imperative(prompt) or len(prompt.split()) < MIN_REPHRASE_WORDS:
        return prompt
    return random.choice(PHRASINGS).format(prompt)
