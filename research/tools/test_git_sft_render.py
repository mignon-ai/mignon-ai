"""
Tests for git SFT template rendering. Run from the repository root:
  tools/.venv/bin/python -m unittest discover -s tools -p "test_*.py"
"""

import unittest

from git_sft_render import PHRASINGS, ground_command, is_imperative, pick, rephrase

PUSH = "git push {remote} {branch}"
VALUES = {"remote": "gitlab", "branch": "dev"}


class TestGroundCommand(unittest.TestCase):
    def test_all_fields_named(self):
        self.assertEqual(ground_command(PUSH, {"remote", "branch"}, VALUES), "git push gitlab dev")

    def test_unnamed_remote_before_named_branch_defaults_to_origin(self):
        self.assertEqual(ground_command(PUSH, {"branch"}, VALUES), "git push origin dev")

    def test_unnamed_trailing_fields_are_dropped(self):
        self.assertEqual(ground_command("git push --force-with-lease {remote} {branch}", set(), VALUES),
                         "git push --force-with-lease")
        self.assertEqual(ground_command(PUSH, {"remote"}, VALUES), "git push gitlab")

    def test_options_after_a_dropped_field_are_kept(self):
        self.assertEqual(ground_command("git push {remote} --tags", set(), VALUES), "git push --tags")

    def test_missing_field_without_default_is_an_error(self):
        with self.assertRaises(ValueError):
            ground_command("git merge {branch} {tag}", {"tag"}, {"branch": "dev", "tag": "v1"})


class TestPick(unittest.TestCase):
    def test_command_never_names_an_argument_missing_from_the_prompt(self):
        prompts = ["push {branch} to {remote}", "push {branch}", "force push safely"]
        for _ in range(50):
            prompt, command = pick(prompts, PUSH, **VALUES)
            for arg in command.split()[2:]:
                self.assertTrue(arg == "origin" or arg in prompt, f"{command!r} invents {arg!r} for {prompt!r}")


class TestRephrase(unittest.TestCase):
    def test_imperative_detection(self):
        self.assertTrue(is_imperative("undo the last commit"))
        for phrased in ("how do I stash?", "please checkout dev", "git status", "what is a rebase"):
            self.assertFalse(is_imperative(phrased), phrased)

    def test_rephrase_wraps_imperatives_only(self):
        wrapped = {p.format("stash all current changes") for p in PHRASINGS}
        for _ in range(20):
            self.assertIn(rephrase("stash all current changes"), wrapped)
        self.assertEqual(rephrase("how do I stash?"), "how do I stash?")
        self.assertEqual(rephrase("stash pop"), "stash pop")


if __name__ == "__main__":
    unittest.main()
