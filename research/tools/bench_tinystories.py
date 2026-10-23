"""
Benchmark pretrained TinyStories models on CPU.

Measures, for each model:
  - total params and non-embedding params
  - bits-per-byte (BPB) on the TinyStories validation set (tokenizer-independent)
  - CPU generation speed (tokens/sec, batch size 1, greedy)

Setup:
    pip install torch transformers datasets

Run:
    python bench_tinystories.py                  # defaults: 1M and 3M, 200 stories
    python bench_tinystories.py --models 1M 3M 8M --n-stories 500 --threads 4
"""

import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinystories_common import load_stories

# TinyStories models use the GPT-Neo tokenizer (per the model cards).
TOKENIZER = "EleutherAI/gpt-neo-125M"
# Context used for bits-per-byte: BOS plus up to MAX_LEN - 1 story tokens.
MAX_LEN = 512


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    # GPT-Neo also has learned position embeddings (transformer.wpe)
    wpe = getattr(getattr(model, "transformer", None), "wpe", None)
    if wpe is not None:
        emb += wpe.weight.numel()
    # Output head is tied to input embeddings in these models, so it isn't double-counted.
    return total, total - emb


def scored_ids(tok, text, max_len=MAX_LEN):
    """Token ids of a story that are scored: truncated so BOS + ids fit in max_len."""
    return tok(text, add_special_tokens=False)["input_ids"][: max_len - 1]


@torch.no_grad()
def bits_per_byte(model, tok, texts, max_len=MAX_LEN):
    """Sum of negative log-likelihood (in bits) divided by number of UTF-8 bytes."""
    total_bits, total_bytes = 0.0, 0
    bos = tok.eos_token_id  # GPT-Neo uses <|endoftext|> as the document separator
    for text in texts:
        ids = scored_ids(tok, text, max_len)
        if not ids:
            continue
        # Count only the bytes we actually scored (in case of truncation)
        scored_text = tok.decode(ids)
        input_ids = torch.tensor([[bos] + ids])
        logits = model(input_ids).logits[0, :-1]          # predictions for ids
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll_nats = -logp.gather(1, torch.tensor(ids)[:, None]).sum().item()
        total_bits += nll_nats / math.log(2)
        total_bytes += len(scored_text.encode("utf-8"))
    return total_bits / total_bytes


@torch.no_grad()
def gen_speed(model, tok, prompt="Once upon a time, there was a little", new_tokens=128, reps=3):
    inputs = tok(prompt, return_tensors="pt")
    kwargs = dict(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    model.generate(**inputs, **kwargs)  # warmup
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = model.generate(**inputs, **kwargs)
        times.append(time.perf_counter() - t0)
    best = min(times)
    sample = tok.decode(out[0], skip_special_tokens=True)
    return new_tokens / best, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["1M", "3M"],
                    help="Sizes from: 1M 3M 8M 28M 33M")
    ap.add_argument("--n-stories", type=int, default=200)
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for torch (default: torch's choice)")
    ap.add_argument("--out", default="baseline_results.json")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch threads: {torch.get_num_threads()}")

    print(f"Loading {args.n_stories} validation stories...")
    texts = list(load_stories("validation", args.n_stories))

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    results = []
    for size in args.models:
        name = f"roneneldan/TinyStories-{size}"
        print(f"\n=== {name} ===")
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
        total, non_emb = count_params(model)

        t0 = time.perf_counter()
        bpb = bits_per_byte(model, tok, texts)
        eval_s = time.perf_counter() - t0

        tps, sample = gen_speed(model, tok)

        r = {
            "model": name,
            "total_params": total,
            "non_embedding_params": non_emb,
            "bits_per_byte": round(bpb, 4),
            "gen_tokens_per_sec": round(tps, 1),
            "eval_seconds": round(eval_s, 1),
        }
        results.append(r)
        print(f"params: {total/1e6:.2f}M total, {non_emb/1e6:.2f}M non-embedding")
        print(f"bits/byte: {bpb:.4f}   (eval took {eval_s:.1f}s)")
        print(f"CPU generation: {tps:.1f} tokens/sec")
        print(f"sample: {sample[:300]!r}")

    print("\n" + "-" * 78)
    print(f"{'model':32} {'total':>8} {'non-emb':>8} {'BPB':>7} {'tok/s':>8}")
    for r in results:
        print(f"{r['model']:32} {r['total_params']/1e6:7.2f}M {r['non_embedding_params']/1e6:7.2f}M "
              f"{r['bits_per_byte']:7.4f} {r['gen_tokens_per_sec']:8.1f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
