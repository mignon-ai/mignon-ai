"""
Write the exact text that bench_tinystories.py scores, so 'mignon eval -per-doc' can score our model
on the same bytes.

Each story is truncated the way the benchmark truncates it (BOS + at most 511 GPT-Neo tokens) and
stories are separated by '<EOS>' lines, which 'mignon eval -per-doc' scores as independent documents.

    python research/tools/dump_tinystories_heldout.py --n-stories 200 --out data/tinystories_heldout.txt
    mignon eval -load models/tiny.json -heldout data/tinystories_heldout.txt -per-doc
"""

import argparse

from transformers import AutoTokenizer

from bench_tinystories import MAX_LEN, TOKENIZER, scored_ids
from tinystories_common import DOC_SEPARATOR, load_stories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stories", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--out", default="tinystories_heldout.txt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    docs = [tok.decode(scored_ids(tok, t, args.max_len)) for t in load_stories("validation", args.n_stories)]
    docs = [d for d in docs if d]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(DOC_SEPARATOR.join(docs))
    print(f"Wrote {len(docs)} stories, {sum(len(d.encode('utf-8')) for d in docs)} bytes to {args.out}")


if __name__ == "__main__":
    main()
