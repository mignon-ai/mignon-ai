#!/usr/bin/env python3
"""
build_tech_corpus.py

Mixes the cleaned sources in data/sources/ into shuffled, sharded pretraining corpora:
  --stage foundation   data/pretrain_tech/   broad simple technical English (train from scratch)
  --stage git          data/pretrain_git/    git-heavy text for a short continued-pretraining pass

Full pipeline (run from the repository root with tools/.venv/bin/python):
  1. research/tools/fetch_tldr.py                tldr.txt, tldr_git_pairs.json
  2. research/tools/fetch_git_docs.py --progit   git_docs.txt, progit.txt
  3. research/tools/fetch_stackexchange.py       stackexchange.txt, stackexchange_git.txt
  4. research/tools/fetch_commit_messages.py     commits.txt
  5. research/tools/fetch_web_tech.py            web_tech.txt
  6. research/tools/generate_tinytech.py         tinytech.txt (Claude Batches API, paid; --backend ollama runs it locally)
  7. research/tools/build_tech_corpus.py --stage foundation   and   --stage git
     (optional: research/tools/fetch_simple_wiki.py, then --stage english for an English-first curriculum)
  8. research/tools/generate_git_sft_extras.py   git_sft_extras.json (Claude Batches API, paid, optional; --backend ollama)
  9. research/tools/augment_git_sft.py           data/git_sft_v2.json + held-out pairs
 10. research/tools/build_git_probes.py          research/eval/git_tasks.json

Then: train on data/pretrain_tech_v3, continue on data/pretrain_git_v3 with the same tokenizer,
fine-tune with `mignon train -sft data/git_sft_v2.json`, and score with `mignon eval -benchmark -tasks research/eval/git_tasks.json`.

Usage: tools/.venv/bin/python research/tools/build_tech_corpus.py --stage foundation [--target-mb 140]
"""

import argparse
import re

from techcorpus_common import (
    DEFAULT_MAX_BLOCK_WORDS, FORBIDDEN_MARKERS, MIN_SHARDS, normalize, read_blocks, seeded_rng,
    source_path, word_count, write_shards,
)

GIT_RE = re.compile(r"\bgit\b", re.IGNORECASE)

# Corpus sizes follow the large preset (~1.95M params): about 20 tokens per parameter at
# ~3.5 bytes per token is ~140 MB for the foundation stage. The git stage is continued
# pretraining and is limited by how much git text exists.
# stage -> [(source file, share of the corpus, only blocks that mention git)]
STAGES = {
    "foundation": {
        "out_dir": "data/pretrain_tech_v3",
        "target_mb": 140.0,
        "mix": [
            ("tinytech.txt", 0.35, False),
            ("stackexchange.txt", 0.20, False),
            ("stackexchange_git.txt", 0.10, False),
            ("commits.txt", 0.12, False),
            ("tldr.txt", 0.05, False),
            ("git_docs.txt", 0.05, False),
            ("progit.txt", 0.03, False),
            ("web_tech.txt", 0.15, False),
        ],
    },
    "git": {
        "out_dir": "data/pretrain_git_v3",
        "target_mb": 35.0,
        "mix": [
            ("git_docs.txt", 0.15, False),
            ("progit.txt", 0.10, False),
            ("stackexchange_git.txt", 0.30, False),
            ("tinytech.txt", 0.15, True),
            ("tldr.txt", 0.10, True),
            ("commits.txt", 0.10, False),
            ("web_tech.txt", 0.10, True),
        ],
    },
    # Optional English-first curriculum stage, trained before "foundation" (fetch_simple_wiki.py).
    "english": {
        "out_dir": "data/pretrain_english",
        "target_mb": 140.0,
        "mix": [
            ("simplewiki.txt", 1.0, False),
        ],
    },
}
BYTES_PER_MB = 1_000_000


def keep_block(block, max_words, ascii_only):
    """Blocks that fit the context, carry no special markers and (optionally) are pure ASCII.

    Rare scripts and emoji each take a tokenizer vocab slot while appearing a handful of times.
    """
    return (word_count(block) <= max_words and not any(m in block for m in FORBIDDEN_MARKERS)
            and (block.isascii() or not ascii_only))


def load_source(name, git_only, max_words, ascii_only):
    try:
        blocks = [normalize(b) for b in read_blocks(source_path(name))]
    except FileNotFoundError:
        return None
    if git_only:
        blocks = [b for b in blocks if GIT_RE.search(b)]
    return [b for b in blocks if keep_block(b, max_words, ascii_only)]


def sample_bytes(blocks, budget, max_repeat, rng):
    """Random blocks up to budget bytes; small sources are repeated at most max_repeat times."""
    pool = list(blocks)
    rng.shuffle(pool)
    picked, used, passes = [], 0, 0
    while used < budget and passes < max_repeat and pool:
        for b in pool:
            if used >= budget:
                break
            picked.append(b)
            used += len(b.encode("utf-8")) + 2
        passes += 1
        rng.shuffle(pool)
    return picked, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--target-mb", type=float, default=0.0, help="corpus size (0 = stage default)")
    ap.add_argument("--max-repeat", type=int, default=2, help="max passes over a source that is too small")
    ap.add_argument("--shards", type=int, default=8, help=f"output files (at least {MIN_SHARDS})")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_BLOCK_WORDS, help="drop longer blocks")
    ap.add_argument("--allow-non-ascii", action="store_true",
                    help="keep blocks with characters that have no ASCII spelling after folding")
    ap.add_argument("--out-dir", default="", help="output directory (default per stage)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stage = STAGES[args.stage]
    rng = seeded_rng(args.seed)
    target = (args.target_mb or stage["target_mb"]) * BYTES_PER_MB

    sources = []
    for name, share, git_only in stage["mix"]:
        blocks = load_source(name, git_only, args.max_words, not args.allow_non_ascii)
        if not blocks:
            print(f"  {name}: missing or empty, skipped (its share goes to the other sources)")
            continue
        sources.append((name, share, blocks))
    if not sources:
        raise SystemExit("No sources found in data/sources/. Run the fetch_*.py scripts first.")

    total_share = sum(share for _, share, _ in sources)
    corpus = []
    for name, share, blocks in sources:
        picked, used = sample_bytes(blocks, target * share / total_share, args.max_repeat, rng)
        corpus.extend(picked)
        print(f"  {name}: {used / BYTES_PER_MB:.1f} MB ({len(picked)} blocks, {share / total_share:.0%} target)")

    rng.shuffle(corpus)
    out_dir = args.out_dir or stage["out_dir"]
    size = write_shards(corpus, out_dir, args.shards)
    print(f"{args.stage} corpus: {len(corpus)} blocks, {size / BYTES_PER_MB:.1f} MB -> {out_dir}/")


if __name__ == "__main__":
    main()
