#!/usr/bin/env python3
"""
generate_off_topic.py

Writes the requests an order-search assistant should refuse ("orders unknown"): small talk, questions
about other topics, commands for other assistants, and actions on orders that a search cannot do. A
language model (by default a local Ollama model) writes them in many wordings, so the model learns
that "not an order search" covers far more than a few phrases. Gibberish is added in code.

Output: a JSON list of request strings, for build_query_sft.py --off-topic.

Usage:
  python3 tools/generate_off_topic.py --out data/exp/off_topic.json --state data/exp/off_topic_state.json
Re-running with the same arguments resumes.
"""

import argparse
import re

import llm_backend
from techcorpus_common import dedupe, is_mostly_ascii, save_json, word_count

PER_REQUEST = 20
MIN_WORDS, MAX_WORDS = 1, 16

# Each kind: what to write, plus subjects that make the lists differ from each other.
KINDS = {
    "smalltalk": ("greetings, thanks, goodbyes, questions about the assistant itself, and casual chat",
                  ["morning greetings", "thanking the assistant", "asking what it can do", "complaining", "jokes",
                   "asking who built it", "goodbyes", "compliments", "being bored", "testing the chat"]),
    "knowledge": ("general knowledge questions a person might ask a search engine",
                  ["geography", "history", "science", "sports results", "celebrities", "animals", "health",
                   "food and cooking", "movies and music", "travel"]),
    "assistant": ("commands for a voice or personal assistant that have nothing to do with orders",
                  ["alarms and timers", "music", "smart-home devices", "calendar and meetings", "reminders",
                   "navigation", "phone calls", "weather", "messages and email", "shopping lists"]),
    "writing": ("requests to write, translate, summarise or calculate something",
                ["poems", "emails to friends", "translations", "maths problems", "unit conversions",
                 "code snippets", "stories", "slogans", "summaries", "spelling help"]),
    "order_actions": ("requests to change something about an order, customer or account, which a read-only "
                      "search tool cannot do (never ask to search, list or count)",
                      ["cancelling an order", "refunding an order", "editing an address", "creating an order",
                       "emailing a customer", "deleting a customer", "changing a price", "printing a label",
                       "tracking one parcel by number", "reordering"]),
    "support": ("customer-support questions about a shop's policies and the person's own account, "
                "not about searching the order database",
                ["returns policy", "shipping costs", "opening hours", "payment methods", "password problems",
                 "delivery delays", "discount codes", "contacting support", "warranty", "gift cards"]),
    "fragments": ("short vague or incomplete messages: single words, filler, half sentences, reactions, or "
                  "messages that refer to something earlier in a conversation that does not exist",
                  ["single words", "filler words", "reactions", "unfinished sentences", "references to earlier",
                   "typos", "questions without context", "numbers only", "emoji-style reactions in words", "random remarks"]),
}

PROMPT = """Write {n} different messages that a person might type into a chat box. Kind: {what}.
Subject for this list: {subject}.

Rules:
- Each message is one line of plain text, between 1 and 14 words, written the way real people type.
- Vary length, tone and wording. Do not number them or repeat them.
- None of them may ask to search, list, find, count or total orders, invoices, purchases or sales."""

# A message that still reads as an order search would teach the model to refuse real requests.
SEARCH_CUE = re.compile(r"\b(unpaid|overdue|invoices?|how many orders|list orders|find orders|show orders|"
                        r"total of|revenue|purchases|transactions)\b", re.IGNORECASE)


def valid(text):
    text = text.strip()
    return (MIN_WORDS <= word_count(text) <= MAX_WORDS and is_mostly_ascii(text) and not SEARCH_CUE.search(text)
            and "{" not in text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--repeats", type=int, default=2, help="lists per kind and subject")
    ap.add_argument("--dry-run", action="store_true")
    llm_backend.add_arguments(ap)
    ap.set_defaults(backend="ollama")
    args = ap.parse_args()
    backend, model, effort = llm_backend.resolve(args)

    schema = backend.json_list_schema("messages", {"type": "string"})
    requests = []
    for kind, (what, subjects) in KINDS.items():
        for si, subject in enumerate(subjects):
            for rep in range(args.repeats):
                requests.append(backend.build_request(
                    f"off-{kind}-{si}-{rep}", PROMPT.format(n=PER_REQUEST, what=what, subject=subject),
                    schema, model=model, effort=effort))
    if args.dry_run:
        backend.dry_run(requests)
        return

    results = backend.run(requests, args.state, "messages")
    messages = [t.strip() for items in results.values() for t in items]
    kept = dedupe([t for t in messages if valid(t)], key=str.lower)
    save_json(args.out, kept)
    print(f"{len(requests)} requests, {len(messages)} messages, {len(kept)} kept -> {args.out}")


if __name__ == "__main__":
    main()
