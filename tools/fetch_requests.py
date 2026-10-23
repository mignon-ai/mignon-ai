#!/usr/bin/env python3
"""
fetch_requests.py

Downloads three public collections of short requests that real people wrote to assistants and writes them as
a pretraining text file, so a base model has seen how people phrase a request before it is fine-tuned on one
task:

  Banking77   13k customer questions to a bank (PolyAI, CC BY 4.0)
  CLINC150    23k requests across 150 everyday intents, with off-topic ones (Larson et al. 2019, CC BY 3.0)
  MASSIVE     English requests to a home assistant: alarms, calendar, shopping lists, music, ... (Amazon, CC BY 4.0)

Only the request text is used, never the labels. Output: blocks separated like the other pretraining files
(see techcorpus_common.write_blocks), in a seeded random order.

Usage: python3 tools/fetch_requests.py [--out data/sources/requests.txt] [--repeat 1]
"""

import argparse
import csv
import io
import json
import tarfile
import urllib.request

from techcorpus_common import dedupe, seeded_rng, write_blocks

BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
CLINC150_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
MASSIVE_ENGLISH = "1.0/data/en-US.jsonl"


def download(url):
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def banking77():
    texts = []
    for split in ("train", "test"):
        rows = csv.DictReader(io.StringIO(download(BANKING77_URL.format(split=split)).decode("utf-8")))
        texts += [r["text"] for r in rows]
    return texts


def clinc150():
    data = json.loads(download(CLINC150_URL))
    return [text for key in data for text, _ in data[key]]


def massive_english():
    with tarfile.open(fileobj=io.BytesIO(download(MASSIVE_URL)), mode="r:gz") as tar:
        lines = tar.extractfile(MASSIVE_ENGLISH).read().decode("utf-8").splitlines()
    return [json.loads(line)["utt"] for line in lines if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sources/requests.txt")
    ap.add_argument("--repeat", type=int, default=1, help="write the whole collection this many times, shuffled each time")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts = []
    for name, fetch in (("Banking77", banking77), ("CLINC150", clinc150), ("MASSIVE", massive_english)):
        got = [" ".join(t.split()) for t in fetch()]
        print(f"  {name}: {len(got)} requests")
        texts += got
    texts = dedupe([t for t in texts if t], key=str.lower)
    rng = seeded_rng(args.seed)
    blocks = []
    for _ in range(args.repeat):
        rng.shuffle(texts)
        blocks += texts
    size = write_blocks(args.out, blocks)
    print(f"{len(texts)} distinct requests, {len(blocks)} blocks, {size / 1e6:.2f} MB -> {args.out}")


if __name__ == "__main__":
    main()
