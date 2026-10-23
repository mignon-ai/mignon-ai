"""
Download the TinyStories train split as mignon training text: stories separated by '<EOS>' lines,
split over several files (mignon reads every .txt file in the -text directory).

    python research/tools/download_tinystories_train.py --n-stories 300000 --out-dir data/tinystories

The validation split used for benchmarking (dump_tinystories_heldout.py) is a separate split, so
nothing here overlaps it.
"""

import argparse
import itertools
import os

from tinystories_common import DOC_SEPARATOR, load_stories

STORIES_PER_FILE = 50_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stories", type=int, default=None, help="Stories to keep (default: all ~2.1M)")
    ap.add_argument("--stories-per-file", type=int, default=STORIES_PER_FILE)
    ap.add_argument("--out-dir", default="data/tinystories")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stories = (s.strip() for s in load_stories("train", args.n_stories))
    stories = (s for s in stories if s)
    written = 0
    for shard in itertools.count():
        chunk = list(itertools.islice(stories, args.stories_per_file))
        if not chunk:
            break
        path = os.path.join(args.out_dir, f"train_{shard:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(DOC_SEPARATOR.join(chunk))
        written += len(chunk)
        print(f"{path}: {len(chunk)} stories")
    print(f"Wrote {written} stories to {args.out_dir}")


if __name__ == "__main__":
    main()
