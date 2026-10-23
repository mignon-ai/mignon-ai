"""Benchmark probes in the format `mignon eval -benchmark` reads: a context to continue and the accepted
continuations. Generators build them from held-out (request, reply) pairs."""


def option_probe(command):
    """(prefix, " option") for the first option of a command that uses -x / --option flags."""
    parts = command.split()
    for i, tok in enumerate(parts[1:], start=1):
        if tok.startswith("-") and len(tok) > 1 and not tok.startswith("--="):
            return " ".join(parts[:i]), " " + tok.split("=", 1)[0]
    return None


def copy_probe(instruction, command):
    """(prefix, " value") for the first argument of the command that the request spells out."""
    parts = command.split()
    for i, tok in enumerate(parts[1:], start=1):
        value = tok.strip("'\"")
        if not tok.startswith("-") and len(value) > 2 and value in instruction:
            return " ".join(parts[:i]), " " + value
    return None


def probe(prompt, expected, instruction="", complete=False):
    """One probe: the model must continue `prompt` (for `instruction`) with one of `expected`.
    `complete` marks the continuation as the whole reply, so the model must also end it."""
    task = {"instruction": instruction, "prompt": prompt, "expected": list(expected)}
    if complete:
        task["complete"] = True
    return task
