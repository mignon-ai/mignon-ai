"""
Generate stories from mignon models and TinyStories models for the same prompts, and write them
shuffled and unlabelled so they can be read and ranked blind.

    python research/tools/compare_story_samples.py \
        --mignon-model models/tinystories_1m_v2.json --hf-model roneneldan/TinyStories-1M

Writes <out-dir>/samples_blind.md (read and rank this) and <out-dir>/samples_key.json (which label
is which model; open it after ranking). Sampling matches mignon's defaults for generate: temperature
0.7, top-p 0.9 and, unless --penalty says otherwise, repetition penalty 1.2.
"""

import argparse
import json
import os
import random
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_tinystories import TOKENIZER

PROMPTS = [
    "Once upon a time",
    "Once upon a time, there was a little dog named Spot.",
    "Tom and Mia went to the park to play.",
    "One day, a girl found a shiny red ball in the garden.",
    "The little bird was sad because",
    "Ben wanted to bake a cake for his mom, but",
    "It was a rainy day, so the children stayed inside.",
    "A big bear lived in the forest. He",
]
TEMPERATURE, DEFAULT_REP_PENALTY, TOP_P = 0.7, 1.2, 0.9
OUTPUT_MARKER = "Output:"
OUTPUT_END = "-----"


def join_prompt(prompt, continuation):
    """Prompt plus continuation. 'mignon generate' trims the continuation's leading space, so a word
    that starts right after a prompt that does not end in whitespace gets its space back. Applied
    to every model alike so the join does not reveal which model wrote a sample."""
    if continuation and not prompt[-1:].isspace() and continuation[0].isalnum():
        continuation = " " + continuation
    return prompt + continuation


def mignon_sample(binary, model, prompt, max_tokens, penalty):
    """Continuation of prompt from 'mignon generate' (the part printed after 'Output:')."""
    out = subprocess.run(
        [binary, "generate", "-load", model, "-prompt", prompt, "-gen-tokens", str(max_tokens),
         "-temp", str(TEMPERATURE), "-penalty", str(penalty), "-top-p", str(TOP_P)],
        capture_output=True, text=True, check=True).stdout
    body = out.split(OUTPUT_MARKER, 1)[1]
    lines = body.rstrip().splitlines()
    if lines and lines[-1].startswith(OUTPUT_END):
        lines = lines[:-1]
    return "\n".join(lines).removeprefix(" ").rstrip()


@torch.no_grad()
def hf_sample(model, tok, prompt, max_tokens, penalty):
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=TEMPERATURE,
                         top_p=TOP_P, repetition_penalty=penalty, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.removeprefix(prompt).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mignon-bin", default="./mignon-ai")
    ap.add_argument("--mignon-model", nargs="+", required=True, help="Saved mignon models to sample")
    ap.add_argument("--hf-model", nargs="+", default=["roneneldan/TinyStories-1M"])
    ap.add_argument("--prompts-file", help="One prompt per line (default: built-in list)")
    ap.add_argument("--samples", type=int, default=3, help="Samples per model and prompt")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--penalty", type=float, default=DEFAULT_REP_PENALTY,
                    help="Repetition penalty for every model (1.0 = off); mignon and Hugging Face apply it differently")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="samples")
    args = ap.parse_args()

    prompts = PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    generators = {}  # model name -> fn(prompt) -> continuation
    for path in args.mignon_model:
        generators[path] = lambda p, path=path: mignon_sample(args.mignon_bin, path, p, args.max_tokens, args.penalty)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    for name in args.hf_model:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
        generators[name] = lambda p, model=model: hf_sample(model, tok, p, args.max_tokens, args.penalty)

    samples = []  # (prompt, model, text)
    for prompt in prompts:
        for name, generate in generators.items():
            for _ in range(args.samples):
                samples.append((prompt, name, join_prompt(prompt, generate(prompt))))
        print(f"done: {prompt!r}")

    rng.shuffle(samples)
    os.makedirs(args.out_dir, exist_ok=True)
    key = {}
    with open(os.path.join(args.out_dir, "samples_blind.md"), "w", encoding="utf-8") as f:
        for i, (prompt, name, text) in enumerate(samples, 1):
            label = f"S{i:03d}"
            key[label] = {"model": name, "prompt": prompt}
            f.write(f"## {label}\n\n_Prompt: {prompt}_\n\n{text}\n\n")
    with open(os.path.join(args.out_dir, "samples_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)
    print(f"Wrote {len(samples)} samples to {args.out_dir}/samples_blind.md (key: samples_key.json)")


if __name__ == "__main__":
    main()
