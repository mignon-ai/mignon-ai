#!/usr/bin/env python3
"""
augment_git_sft.py

Merges every git SFT source into one rebalanced dataset and splits off held-out *intents*:
  inputs   data/git_sft.json                    templates (build_git_sft.py)
           data/sources/tldr_git_pairs.json     tldr examples (fetch_tldr.py)
           data/sources/git_sft_extras.json     Claude extras (generate_git_sft_extras.py), optional
           data/sources/tldr.txt                general CLI replay, filtered against held-out intents and eval
           research/tools/git_glossary.py                hand-written flag glossary, repeated --glossary-repeat times
  outputs  data/git_sft_v2.json                 training set for `mignon train -sft`
           data/git_sft_v2_heldout.json         evaluation pairs for build_git_probes.py, each tagged with
                                                split "unseen_intent" (command form never trained) or
                                                "seen_intent" (trained command form, unseen request)

The most common intent of every subcommand always stays in training, so held-out intents are
variants (new flags) of commands the model knows, never a whole command it could not know.
Each intent (command skeleton, see git_intents.py) is capped so that a few templated commands
(git config user.name, git push) no longer dominate the gradient.

Usage: tools/.venv/bin/python research/tools/augment_git_sft.py [--max-per-intent 500] [--heldout 0.1] [--heldout-seen 0.02] [--tech-replay 2500]
"""

import argparse
import collections
import os

import git_glossary
from git_intents import intent_key, is_heldout, subcommand
from techcorpus_common import dedupe, load_json, save_json, seeded_rng, source_path

INPUTS = ["data/git_sft.json", source_path("tldr_git_pairs.json"), source_path("git_sft_extras.json")]
TRAIN_OUT = "data/git_sft_v2.json"
HELDOUT_OUT = "data/git_sft_v2_heldout.json"


def load_inputs(paths):
    pairs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  {path}: missing, skipped")
            continue
        rows = load_json(path)
        pairs.extend({"instruction": r["instruction"].strip(), "output": r["output"].strip()} for r in rows)
        print(f"  {path}: {len(rows)} pairs")
    return dedupe(pairs, key=lambda p: (p["instruction"].lower(), p["output"]))


def cap_per_intent(groups, max_per_intent, rng):
    capped = {}
    for key, items in groups.items():
        items = list(items)
        rng.shuffle(items)
        capped[key] = items[:max_per_intent]
    return capped


def base_intents(groups):
    """The largest intent of each subcommand: the plain form of the command."""
    best = {}
    for key, items in groups.items():
        sub = subcommand(key)
        if sub and len(items) > len(groups.get(best.get(sub), [])):
            best[sub] = key
    return set(best.values())


EVAL_TASKS = "research/eval/git_tasks.json"
MIN_TECH_REPLAY = 500

# Greeting and identity replies mapped to several phrasings of the request, so the greeting prior
# generalizes past the literal strings instead of memorizing one prompt per reply.
CHAT_REPLAYS = {
    "Hello! How can I help you with Git or your codebase today?": ["hello", "hello there", "hello!"],
    "Hi there! What command or workflow do you need help with?": ["hi", "hi there", "hi!"],
    "Hey! Ready to help with Git. What are you looking to do?": ["hey", "hey there", "hey!"],
    "Good morning! How can I assist with your repository today?": ["good morning", "morning"],
    "Good evening! What can I help you with?": ["good evening", "evening"],
    "I am Mignon-AI, your Git and developer command assistant.": [
        "who are you?", "who are you", "what are you?", "introduce yourself"],
    "My name is Mignon-AI!": ["what is your name?", "what's your name?", "your name?"],
    "I can translate natural language requests into exact Git commands, explain Git workflows, and help manage "
    "your branches and commits.": ["what can you do?", "what do you do?", "how can you help me?"],
    "Ask me anything about Git! For example: 'how to push to origin', 'undo last commit', or "
    "'stage changes in a file'.": ["help", "help me", "i need help"],
    "You're very welcome! Let me know if you need anything else.": ["thank you", "thanks a lot", "thank you!"],
    "Glad to help! Happy coding.": ["thanks", "thx", "thanks!"],
    "Goodbye! Have a great time coding.": ["bye", "bye!", "see you"],
    "Goodbye! Let me know whenever you need more Git assistance.": ["goodbye", "goodbye!", "see you later"],
}
CHAT_REPEATS = 10


def pair_key(p):
    return (p["instruction"].lower(), p["output"])


def chat_replay_pairs():
    return [{"instruction": q, "output": a}
            for _ in range(CHAT_REPEATS) for a, qs in CHAT_REPLAYS.items() for q in qs]


def load_eval_instructions(path):
    """Instructions of the eval tasks, lowercased, so replay data can never contain them."""
    if not os.path.exists(path):
        print(f"  {path}: missing, eval leakage check skipped")
        return set()
    return {t["instruction"].strip().lower() for t in load_json(path) if t.get("instruction")}


def parse_tldr(path):
    """tldr.txt blocks are exactly: header line, 'description:' line, command line."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: fetch the tldr source or pass --tech-replay 0")
    with open(path, "r", encoding="utf-8") as f:
        blocks = [b for b in f.read().split("\n\n") if b.strip()]
    samples = []
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) != 3:
            continue
        _, desc, cmd = lines
        if desc.endswith(":") and len(cmd) > 2 and not cmd.startswith("#"):
            samples.append({"instruction": desc.rstrip(":"), "output": cmd})
    return samples


def tldr_replay_pairs(path, rng, count, heldout_keys, seen, eval_instructions):
    """General CLI pairs that keep technical comprehension, minus anything that would leak into the
    held-out or eval sets or duplicate an existing training pair."""
    if count <= 0:
        return []
    kept = []
    for s in parse_tldr(path):
        key = pair_key(s)
        if key in seen or s["instruction"].lower() in eval_instructions or intent_key(s) in heldout_keys:
            continue
        seen.add(key)
        kept.append(s)
    rng.shuffle(kept)
    kept = kept[:count]
    if len(kept) < min(count, MIN_TECH_REPLAY):
        raise SystemExit(f"only {len(kept)} usable pairs in {path}: the tldr format may have changed")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-intent", type=int, default=500, help="max examples per command skeleton")
    ap.add_argument("--heldout", type=float, default=0.1, help="share of intents held out from training")
    ap.add_argument("--heldout-seen", type=float, default=0.02,
                    help="share of examples of trained intents held out (new requests for known commands)")
    ap.add_argument("--tech-replay", type=int, default=2500,
                    help="general CLI pairs from tldr.txt mixed into training (0 disables)")
    ap.add_argument("--glossary-repeat", type=int, default=10,
                    help="copies of the git_glossary.py training pairs mixed into training (0 disables)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = seeded_rng(args.seed)

    pairs = load_inputs(INPUTS)
    groups = collections.defaultdict(list)
    for p in pairs:
        groups[intent_key(p)].append(p)
    groups = cap_per_intent(groups, args.max_per_intent, rng)

    protected = base_intents(groups)
    train, heldout, heldout_intents = [], [], 0
    for key, items in groups.items():
        if key not in protected and is_heldout(key, args.heldout):
            heldout.extend(dict(p, split="unseen_intent") for p in items)
            heldout_intents += 1
            continue
        for p in items:
            if rng.random() < args.heldout_seen:
                heldout.append(dict(p, split="seen_intent"))
            else:
                train.append(p)

    eval_instructions = load_eval_instructions(EVAL_TASKS)
    before = len(train)
    train = [p for p in train if p["instruction"].lower() not in eval_instructions]
    print(f"Dropped {before - len(train)} training pairs whose request also appears in {EVAL_TASKS}")

    heldout_keys = {intent_key(p) for p in heldout if p["split"] == "unseen_intent"}
    seen = {pair_key(p) for p in train} | {pair_key(p) for p in heldout}

    chat = chat_replay_pairs()
    train.extend(chat)
    print(f"Added {len(chat)} chat replay pairs")

    tech = tldr_replay_pairs(source_path("tldr.txt"), rng, args.tech_replay, heldout_keys, seen, eval_instructions)
    train.extend(tech)
    print(f"Added {len(tech)} general CLI replay pairs from tldr.txt")

    if args.glossary_repeat > 0:
        glossary, _, _ = git_glossary.build(EVAL_TASKS)
        glossary = [p for p in glossary if p["instruction"].lower() not in eval_instructions]
        train.extend(glossary * args.glossary_repeat)
        print(f"Added {len(glossary)} glossary pairs x {args.glossary_repeat}")

    rng.shuffle(train)

    save_json(TRAIN_OUT, train)
    save_json(HELDOUT_OUT, heldout)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"{len(pairs)} unique pairs, {len(groups)} intents (largest after cap: {sizes[:5]})")
    print(f"train {len(train)} -> {TRAIN_OUT}")
    print(f"heldout {len(heldout)} from {heldout_intents} intents -> {HELDOUT_OUT}")


if __name__ == "__main__":
    main()
