"""
techcorpus_common.py

Shared helpers for building the technical-English pretraining corpus and the git SFT data.

Block format (matches the Go loader in data.go):
  * One training "block" per paragraph, blocks separated by a single blank line ("\n\n").
    The loader packs whole blocks into a context window and never cuts one in half, so every
    block must stay shorter than the model's context.
  * Inside a block use single newlines only.
  * Documents are separated by a DOC_SEPARATOR line between blank lines. The loader ends every
    document (and every file) with the EOS token. Every block written here is its own document:
    blocks are independent snippets, and write_shards scatters a source's blocks across shards.
  * Never write end-of-sequence markers inside block text; normalize() strips them.
"""

import hashlib
import html
import json
import os
import random
import re
import unicodedata

SOURCES_DIR = "data/sources"

# A block must fit the context window. Presets use 64-128 tokens and a 1-4k BPE vocab, where one
# English word costs roughly 1.3-2 tokens, so ~60 words is the safe default for a 128-token context.
DEFAULT_MAX_BLOCK_WORDS = 60
DEFAULT_MIN_BLOCK_WORDS = 4

# data.go only splits validation by whole files when there are at least this many
# (MinDocsForDocSplit in constants.go). Writing more shards keeps every source in both splits.
MIN_SHARDS = 5

# Markers that must never appear in corpus text: the old "<|eos|>" is not a special token in
# this tokenizer and gets learned as literal characters.
FORBIDDEN_MARKERS = ("<|eos|>", "<EOS>", "<|endoftext|>", "<USER>", "<BOT>", "<PAD>", "<UNK>")

# Ends a document inside a corpus file (DocSeparator in internal/data/data.go). It is the
# tokenizer's EOS spelling, but only ever written as a block of its own by write_blocks.
DOC_SEPARATOR = "<EOS>"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def source_path(name):
    """Path of a cleaned source file inside SOURCES_DIR."""
    return os.path.join(SOURCES_DIR, name)


def strip_markers(text):
    for m in FORBIDDEN_MARKERS:
        text = text.replace(m, " ")
    return text


def html_to_text(raw):
    """Convert Stack Exchange style HTML to plain text, keeping code blocks on their own lines."""
    raw = re.sub(r"(?is)<pre[^>]*>(.*?)</pre>", lambda m: "\n" + m.group(1) + "\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", raw)
    raw = re.sub(r"(?i)<li[^>]*>", "\n- ", raw)
    return html.unescape(_TAG_RE.sub("", raw))


# Typographic characters with a plain ASCII spelling that NFKD does not provide. Every distinct
# character in the tokenizer sample costs a vocab slot, and curly quotes would make the BPE learn
# every contraction twice.
_ASCII_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2022": "-", "\u00b7": "-", "\u2192": "->", "\u2190": "<-", "\u21d2": "=>", "\u00d7": "x",
    "\u200b": "", "\ufeff": "", "\u00ad": "",
})


def fold_to_ascii(text):
    """Map typographic punctuation, odd spaces and accented Latin letters to plain ASCII.

    Characters with no ASCII equivalent (CJK, emoji, ...) are left in place, so callers can
    detect and drop such text with str.isascii().
    """
    text = unicodedata.normalize("NFKD", text.translate(_ASCII_FOLD))
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize(text):
    """Fold to ASCII, collapse whitespace inside lines, drop empty lines, remove forbidden markers."""
    text = fold_to_ascii(strip_markers(text))
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def word_count(text):
    return len(text.split())


def is_mostly_ascii(text, min_ratio=0.97):
    if not text:
        return False
    return sum(1 for c in text if ord(c) < 128) / len(text) >= min_ratio


# Prose heuristics: keyword-stuffed SEO text has many commas and few sentence endings.
MAX_COMMA_RATIO = 0.12
MAX_WORDS_PER_SENTENCE = 35


def looks_like_prose(block):
    words = word_count(block)
    if words == 0:
        return False
    sentences = max(1, len(re.findall(r"[.!?](?:\s|$)", block)))
    return block.count(",") / words <= MAX_COMMA_RATIO and words / sentences <= MAX_WORDS_PER_SENTENCE


def chunk_paragraph(paragraph, max_words=DEFAULT_MAX_BLOCK_WORDS):
    """Split one paragraph into blocks of at most max_words, on sentence boundaries where possible."""
    paragraph = normalize(paragraph)
    if not paragraph:
        return []
    if word_count(paragraph) <= max_words:
        return [paragraph]
    blocks, current = [], []
    for sentence in _SENTENCE_RE.split(paragraph.replace("\n", " ")):
        words = sentence.split()
        # A single over-long sentence is cut on word boundaries as a last resort.
        while len(words) > max_words:
            if current:
                blocks.append(" ".join(current))
                current = []
            blocks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(current) + len(words) > max_words and current:
            blocks.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        blocks.append(" ".join(current))
    return blocks


# Lines starting like this are commands or list items and keep their own line when unwrapping.
_STANDALONE_LINE_RE = re.compile(r"^(?:git |\$ |- |\* |\d+\. )")
_COMMAND_LINE_RE = re.compile(r"^(?:git |\$ )")


def unwrap_lines(paragraph):
    """Join hard-wrapped prose lines; keep commands, list items and 'term:' lines separate."""
    out = []
    for line in paragraph.splitlines():
        line = line.strip()
        if not line:
            continue
        if out and not out[-1].endswith(":") and not _STANDALONE_LINE_RE.match(line) \
                and not _COMMAND_LINE_RE.match(out[-1]):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def text_to_blocks(text, max_words=DEFAULT_MAX_BLOCK_WORDS, min_words=DEFAULT_MIN_BLOCK_WORDS):
    """Split free text on blank lines into context-sized blocks."""
    blocks = []
    for para in re.split(r"\n\s*\n", text):
        blocks.extend(b for b in chunk_paragraph(para, max_words) if word_count(b) >= min_words)
    return blocks


# Tokens longer than this are keys, hashes, base64 or minified code: noise for a small BPE vocab.
MAX_TOKEN_CHARS = 40


def has_junk_token(line):
    return any(len(tok) > MAX_TOKEN_CHARS for tok in line.split())


def take_paragraphs(text, word_budget):
    """Leading whole lines of text that fit word_budget, joined with single newlines.

    Lines with junk tokens are skipped, and a trailing line that introduces something that
    did not fit ("For example:") is dropped.
    """
    kept, used = [], 0
    for line in normalize(text).split("\n"):
        if has_junk_token(line):
            continue
        n = word_count(line)
        if used + n > word_budget:
            break
        kept.append(line)
        used += n
    while kept and kept[-1].endswith(":"):
        kept.pop()
    return "\n".join(kept)


def write_blocks(path, blocks):
    """Write each block as its own document, separated by DOC_SEPARATOR. Returns bytes written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clean = [normalize(b) for b in blocks]
    data = f"\n\n{DOC_SEPARATOR}\n\n".join(b for b in clean if b) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return len(data.encode("utf-8"))


def read_blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        blocks = (b.strip() for b in f.read().split("\n\n"))
        return [b for b in blocks if b and b != DOC_SEPARATOR]


def stable_hash(text, salt=""):
    """Hash that is identical across runs and machines (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.sha256(f"{salt}:{text}".encode()).digest()[:8], "big")


HASH_RANGE = 2**64


def write_shards(blocks, out_dir, n_shards, prefix="part"):
    """Distribute already-shuffled blocks over n_shards files by content hash.

    Copies of a block always land in the same shard, so the shard the Go loader holds out for
    validation never shares text with the training shards. With the same shard count, a block
    also lands in the same shard in every corpus built from these sources.
    """
    n_shards = max(n_shards, MIN_SHARDS)
    os.makedirs(out_dir, exist_ok=True)
    shards = [[] for _ in range(n_shards)]
    for b in blocks:
        shards[stable_hash(normalize(b)) % n_shards].append(b)
    total = 0
    for i, shard in enumerate(shards):
        total += write_blocks(os.path.join(out_dir, f"{prefix}_{i:02d}.txt"), shard)
    return total


def dedupe(items, key=lambda x: x):
    seen, out = set(), []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seeded_rng(seed):
    return random.Random(seed)
