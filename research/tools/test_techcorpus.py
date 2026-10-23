"""
Tests for the dataset tooling. Run from the repository root:
  tools/.venv/bin/python -m unittest discover -s tools -p "test_*.py"
"""

import os
import tempfile
import unittest

import fetch_tldr
from build_git_probes import copy_probe, option_probe
from fetch_git_docs import asciidoc_to_text
from git_intents import command_skeleton, intent_key, is_heldout
from build_tech_corpus import keep_block
from techcorpus_common import (
    DOC_SEPARATOR, MIN_SHARDS, chunk_paragraph, fold_to_ascii, html_to_text, looks_like_prose, normalize, read_blocks,
    seeded_rng, take_paragraphs, text_to_blocks, unwrap_lines, word_count, write_blocks, write_shards,
)


class TestBlocks(unittest.TestCase):
    def test_chunk_respects_word_limit(self):
        para = " ".join(f"Sentence number {i} is here." for i in range(40))
        blocks = chunk_paragraph(para, max_words=20)
        self.assertTrue(all(word_count(b) <= 20 for b in blocks))
        self.assertEqual(sum(word_count(b) for b in blocks), word_count(para))

    def test_overlong_sentence_is_cut(self):
        blocks = chunk_paragraph("word " * 50, max_words=20)
        self.assertEqual([word_count(b) for b in blocks], [20, 20, 10])

    def test_text_to_blocks_drops_tiny_paragraphs(self):
        self.assertEqual(text_to_blocks("ok\n\nThis one has enough words in it.", min_words=4),
                         ["This one has enough words in it."])

    def test_write_blocks_strips_markers_and_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.txt")
            write_blocks(path, ["first block <|eos|>", "second\nblock"])
            self.assertEqual(read_blocks(path), ["first block", "second\nblock"])

    def test_write_blocks_separates_documents(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.txt")
            write_blocks(path, ["first", "second\nblock"])
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), f"first\n\n{DOC_SEPARATOR}\n\nsecond\nblock\n")

    def test_write_shards_enforces_minimum(self):
        with tempfile.TemporaryDirectory() as d:
            write_shards([f"block {i}" for i in range(20)], d, n_shards=2)
            self.assertEqual(len(os.listdir(d)), MIN_SHARDS)

    def test_write_shards_keeps_copies_in_one_shard(self):
        blocks = [f"block number {i}" for i in range(50)] * 2
        with tempfile.TemporaryDirectory() as d:
            write_shards(blocks, d, n_shards=MIN_SHARDS)
            shards = [set(read_blocks(os.path.join(d, f))) for f in sorted(os.listdir(d))]
        for i, a in enumerate(shards):
            for b in shards[i + 1:]:
                self.assertFalse(a & b)
        self.assertEqual(sum(len(s) for s in shards), 50)


class TestTextCleanup(unittest.TestCase):
    def test_fold_to_ascii(self):
        self.assertEqual(fold_to_ascii("It\u2019s a \u201cfix\u201d \u2014 caf\u00e9\u2026"), 'It\'s a "fix" - cafe...')
        self.assertEqual(normalize("a\u00a0b"), "a b")
        self.assertFalse(fold_to_ascii("\u4e2d\u6587 \U0001f680").isascii())

    def test_keep_block_drops_non_ascii_only_when_asked(self):
        self.assertTrue(keep_block("plain text", 60, ascii_only=True))
        self.assertFalse(keep_block("emoji \U0001f680", 60, ascii_only=True))
        self.assertTrue(keep_block("emoji \U0001f680", 60, ascii_only=False))
        self.assertFalse(keep_block("x <EOS>", 60, ascii_only=False))

    def test_unwrap_joins_prose_but_keeps_commands(self):
        text = "This is a hard\nwrapped sentence.\ngit status\nNext line:\nkept apart"
        self.assertEqual(unwrap_lines(text),
                         "This is a hard wrapped sentence.\ngit status\nNext line:\nkept apart")

    def test_take_paragraphs_skips_junk_and_dangling_intro(self):
        text = "Short intro line.\n" + "x" * 80 + "\nFor example:"
        self.assertEqual(take_paragraphs(text, 50), "Short intro line.")

    def test_html_to_text_keeps_code(self):
        self.assertIn("git log", html_to_text("<p>Try</p><pre><code>git log</code></pre>"))

    def test_looks_like_prose(self):
        self.assertTrue(looks_like_prose("Git stores history. Each commit has a parent."))
        self.assertFalse(looks_like_prose("audio, music, sound, plugin, effects, capture, image, screen"))

    def test_asciidoc_drops_options_and_keeps_listing_commands(self):
        src = "NAME\n----\ngit-x - do x\n\nOPTIONS\n-------\n--foo::\n\tDo foo.\n\nEXAMPLES\n--------\n" \
              "Run it:\n------------\n$ git x --bar\noutput line\n------------\n"
        text = asciidoc_to_text(src, keep_options=False)
        self.assertNotIn("Do foo", text)
        self.assertIn("git x --bar", text)
        self.assertNotIn("output line", text)
        self.assertNotIn("$ ", text)


class TestIntents(unittest.TestCase):
    def test_skeleton_removes_entities(self):
        self.assertEqual(command_skeleton("git push -u origin feature/x"), "git push -u <arg>")
        self.assertEqual(command_skeleton("git log --max-count=5"), "git log --max-count")
        self.assertEqual(command_skeleton("ls -la"), "")

    def test_intent_key_for_chains_and_explanations(self):
        self.assertEqual(intent_key({"instruction": "x", "output": "git add . && git commit -m 'm'"}),
                         "git add <arg> && git commit -m <arg>")
        self.assertEqual(intent_key({"instruction": "what does git stash -u do?", "output": "It saves..."}),
                         "about: git stash -u")

    def test_heldout_is_deterministic(self):
        self.assertEqual(is_heldout("git stash -u", 0.5), is_heldout("git stash -u", 0.5))
        self.assertFalse(is_heldout("anything", 0.0))


class TestTldr(unittest.TestCase):
    def test_fill_copies_entities_and_handles_options(self):
        rng = seeded_rng(1)
        cmd, ents = fetch_tldr.fill("git log {{[-p|--patch]}} {{-number_of_commits}}", rng)
        self.assertRegex(cmd, r"^git log (-p|--patch) -\d+$")
        self.assertEqual(ents, [cmd.split()[-1]])  # option choices are not copied, values are
        cmd, ents = fetch_tldr.fill("git switch {{branch_name}}", rng)
        self.assertEqual(cmd, f"git switch {ents[0]}")

    def test_instruction_contains_entities(self):
        self.assertEqual(fetch_tldr.to_instruction("Switch to a branch", ["dev"]), "switch to a branch: dev")

    def test_valid_git_pair(self):
        self.assertTrue(fetch_tldr.valid_git_pair("git-am", "git am patch.diff"))
        self.assertFalse(fetch_tldr.valid_git_pair("git-am", "curl -L x | git am"))
        self.assertFalse(fetch_tldr.valid_git_pair("gitk", "gitk --all"))


class TestProbes(unittest.TestCase):
    def test_option_probe(self):
        self.assertEqual(option_probe("git stash -u"), ("git stash", " -u"))
        self.assertIsNone(option_probe("git switch dev -f"))

    def test_copy_probe_handles_quotes(self):
        self.assertEqual(copy_probe("set name to bob", "git config user.name 'bob'"),
                         ("git config user.name '", "bob"))
        self.assertEqual(copy_probe("switch to dev", "git switch dev"), ("git switch", " dev"))
        self.assertIsNone(copy_probe("push", "git push origin main"))


if __name__ == "__main__":
    unittest.main()
