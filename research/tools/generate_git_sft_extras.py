#!/usr/bin/env python3
"""
generate_git_sft_extras.py

Uses Claude (Message Batches API) to add the git SFT examples that the templates in build_git_sft.py
cannot produce:
  paraphrase  new phrasings of existing requests (casual, terse, typos, questions), same command
  explain     "what does git X do?" -> one or two plain sentences
  error_fix   a git error or problem -> the command(s) that fix it
  multi_step  a goal that needs several commands -> commands joined with &&

Output: data/sources/git_sft_extras.json ({"instruction", "output", "kind"})

Usage:
  tools/.venv/bin/python research/tools/generate_git_sft_extras.py --dry-run
  tools/.venv/bin/python research/tools/generate_git_sft_extras.py --paraphrase 600 --explain 120 --error-fix 150 --multi-step 150
"""

import argparse

import claude_batch
import llm_backend
from git_intents import intent_key
from techcorpus_common import dedupe, load_json, save_json, seeded_rng, source_path, word_count

BASE_SFT = "data/git_sft.json"
TLDR_PAIRS = source_path("tldr_git_pairs.json")
EXAMPLES_PER_REQUEST = 10
PARAPHRASES_PER_EXAMPLE = 5
PAIRS_PER_REQUEST = 10
MAX_EXPLANATION_WORDS = 40

ERROR_THEMES = [
    "push rejected because the remote has new commits", "merge conflict after pull",
    "detached HEAD state", "local changes would be overwritten by checkout",
    "not a git repository", "pathspec did not match any files", "refusing to merge unrelated histories",
    "permission denied (publickey) when pushing", "nothing to commit", "divergent branches on pull",
    "committed to the wrong branch", "committed a file that should be ignored",
    "wrong author name or email on commits", "accidentally deleted a branch",
    "a rebase stopped with conflicts", "a file is too large to push", "the remote URL is wrong",
    "no upstream branch is set", "cannot delete the branch you are on", "untracked files block a merge",
]

GOAL_THEMES = [
    "starting a new feature", "saving all work and sharing it", "updating a branch from main",
    "fixing the last commit", "cleaning up after a merged branch", "setting up a new repository",
    "releasing a version", "moving work to another branch", "syncing a fork with upstream",
    "starting over from the remote state", "configuring identity on a new machine",
    "inspecting what changed before committing",
]

PAIR_SCHEMA = claude_batch.json_list_schema("pairs", {
    "type": "object",
    "properties": {"instruction": {"type": "string"}, "output": {"type": "string"}},
    "required": ["instruction", "output"],
    "additionalProperties": False,
})

COMMON_RULES = """Rules:
- Instructions sound like real people typing to an assistant: vary length, tone and wording. Some are questions, some are commands, a few have small typos.
- Use realistic, varied names for branches, files, remotes, tags and people.
- Commands must be correct and complete, written exactly as typed, using only standard git."""

PARAPHRASE_PROMPT = """Here are requests a user sent to a git assistant, each with the correct command:

{examples}

For each request, write {k} new ways a user could ask for exactly the same thing. Keep every name, path, hash, URL and message exactly as given, and keep the command unchanged in "output".

""" + COMMON_RULES

EXPLAIN_PROMPT = """For each git command below, write one pair where the instruction asks what the command does (vary the wording) and the output explains it in one or two short, plain sentences of at most {max_words} words. No markdown.

{commands}

""" + COMMON_RULES

ERROR_FIX_PROMPT = """Write {n} different pairs about this git problem: {theme}.
The instruction describes the situation or pastes the error message git printed, as a user would. The output is only the git command or commands that fix it, joined with && when there are several.

""" + COMMON_RULES

MULTI_STEP_PROMPT = """Write {n} different pairs about this git goal: {theme}.
The instruction asks for something that needs two to four git commands. The output is only those commands, in order, joined with &&.

""" + COMMON_RULES


def representative_examples(pairs, rng):
    """One example per intent, so paraphrases spread over all commands instead of the most common ones."""
    by_intent = {}
    for p in pairs:
        by_intent.setdefault(intent_key(p), []).append(p)
    reps = [rng.choice(group) for group in by_intent.values()]
    rng.shuffle(reps)
    return reps


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_requests(args, rng, backend):
    base = load_json(BASE_SFT) + load_json(TLDR_PAIRS)
    reps = representative_examples(base, rng)
    prompts = {}

    for i, group in enumerate(chunks((reps * args.paraphrase)[:args.paraphrase * EXAMPLES_PER_REQUEST],
                                     EXAMPLES_PER_REQUEST)[:args.paraphrase]):
        examples = "\n".join(f'- "{p["instruction"]}" -> {p["output"]}' for p in group)
        prompts[f"paraphrase-{i}"] = PARAPHRASE_PROMPT.format(examples=examples, k=PARAPHRASES_PER_EXAMPLE)

    commands = dedupe([p["output"] for p in reps if p["output"].startswith("git ")])
    for i, group in enumerate(chunks(commands, PAIRS_PER_REQUEST)[:args.explain]):
        prompts[f"explain-{i}"] = EXPLAIN_PROMPT.format(
            commands="\n".join(f"- {c}" for c in group), max_words=MAX_EXPLANATION_WORDS)

    for i in range(args.error_fix):
        prompts[f"error_fix-{i}"] = ERROR_FIX_PROMPT.format(n=PAIRS_PER_REQUEST, theme=rng.choice(ERROR_THEMES))
    for i in range(args.multi_step):
        prompts[f"multi_step-{i}"] = MULTI_STEP_PROMPT.format(n=PAIRS_PER_REQUEST, theme=rng.choice(GOAL_THEMES))

    return [backend.build_request(cid, prompt, PAIR_SCHEMA, model=args.model, effort=args.effort)
            for cid, prompt in prompts.items()]


def valid_commands(output):
    return all(part.strip().startswith("git ") for part in output.split("&&"))


def validate(kind, pair):
    inst, out = pair["instruction"].strip(), pair["output"].strip()
    if not inst or not out or "\n" in out:
        return False
    if kind == "explain":
        return word_count(out) <= MAX_EXPLANATION_WORDS and not out.startswith("git ")
    return valid_commands(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paraphrase", type=int, default=600, help=f"requests of {EXAMPLES_PER_REQUEST} examples")
    ap.add_argument("--explain", type=int, default=120, help=f"requests of {PAIRS_PER_REQUEST} commands")
    ap.add_argument("--error-fix", type=int, default=150, help=f"requests of {PAIRS_PER_REQUEST} pairs")
    ap.add_argument("--multi-step", type=int, default=150, help=f"requests of {PAIRS_PER_REQUEST} pairs")
    llm_backend.add_arguments(ap)
    ap.add_argument("--state", default=source_path("git_sft_extras_batch.json"), help="batch id file for resuming")
    ap.add_argument("--dry-run", action="store_true", help="print example prompts and exit without calling the API")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    backend, args.model, args.effort = llm_backend.resolve(args)

    requests = build_requests(args, seeded_rng(args.seed), backend)
    if args.dry_run:
        kinds = {}
        for r in requests:
            kinds.setdefault(r["custom_id"].rsplit("-", 1)[0], r)
        backend.dry_run(list(kinds.values()), show=len(kinds))
        return

    results = backend.run(requests, args.state, "pairs")
    extras, rejected = [], 0
    for cid, pairs in results.items():
        kind = cid.rsplit("-", 1)[0]
        for p in pairs:
            if validate(kind, p):
                extras.append({"instruction": p["instruction"].strip(), "output": p["output"].strip(), "kind": kind})
            else:
                rejected += 1
    extras = dedupe(extras, key=lambda p: (p["instruction"].lower(), p["output"]))
    save_json(source_path("git_sft_extras.json"), extras)
    print(f"git_sft_extras -> {len(extras)} pairs ({rejected} rejected by validation)")


if __name__ == "__main__":
    main()
