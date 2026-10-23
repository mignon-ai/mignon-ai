import random
import unittest

from generate_paraphrases import sample_pairs, valid_paraphrase, value_tokens

INST = "copy report.txt to backup.sql"
CMD = "cp report.txt backup.sql"


class ParaphraseTest(unittest.TestCase):
    def test_value_tokens_are_the_arguments_named_in_the_request(self):
        self.assertEqual(value_tokens(INST, CMD), ["report.txt", "backup.sql"])
        self.assertEqual(value_tokens("list files", "ls -l"), [])

    def test_keeps_values_and_rewords(self):
        self.assertTrue(valid_paraphrase("Can you make a copy of report.txt called backup.sql?", INST, CMD))

    def test_rejects_lost_value(self):
        self.assertFalse(valid_paraphrase("Can you make a copy of the report?", INST, CMD))

    def test_rejects_copy_of_the_request_the_command_code_and_chatter(self):
        self.assertFalse(valid_paraphrase(INST.upper(), INST, CMD))
        self.assertFalse(valid_paraphrase("Please run cp report.txt backup.sql for me", INST, CMD))
        self.assertFalse(valid_paraphrase("Use `report.txt` to make backup.sql", INST, CMD))
        self.assertFalse(valid_paraphrase("Here are some ways to ask for report.txt and backup.sql", INST, CMD))

    def test_rejects_too_short_and_too_long(self):
        self.assertFalse(valid_paraphrase("report.txt backup.sql", INST, CMD))
        self.assertFalse(valid_paraphrase("please copy report.txt to backup.sql " + "now " * 40, INST, CMD))

    def test_rejects_paraphrase_much_longer_than_source(self):
        long_text = "Can you make a copy of report.txt called backup.sql and also tell me everything it does to the disk?"
        self.assertFalse(valid_paraphrase(long_text, INST, CMD))
        self.assertTrue(valid_paraphrase(long_text, "copy report.txt to backup.sql and tell me what it does to the disk", CMD))

    def test_only_names_lets_plain_words_change(self):
        inst, cmd = "find unpaid orders from Acme over 500", "orders find --status unpaid --customer Acme --min-amount 500"
        text = "which of Acme's outstanding invoices are above 500"
        self.assertFalse(valid_paraphrase(text, inst, cmd))
        self.assertTrue(valid_paraphrase("which outstanding invoices from Acme are above 500", inst, cmd, only_names=True))
        self.assertFalse(valid_paraphrase("which outstanding invoices from Acme are above 400", inst, cmd, only_names=True))

    def test_sample_spreads_over_tools(self):
        pairs = [{"instruction": f"i{i}", "output": f"ls {i}"} for i in range(30)]
        pairs += [{"instruction": f"j{i}", "output": f"cp {i}"} for i in range(5)]
        got = sample_pairs(pairs, 3, random.Random(1))
        self.assertEqual(sorted(p["output"].split()[0] for p in got), ["cp"] * 3 + ["ls"] * 3)


if __name__ == "__main__":
    unittest.main()
