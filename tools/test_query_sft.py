import unittest

import build_query_sft as b
from techcorpus_common import seeded_rng


class QuerySftTest(unittest.TestCase):
    def setUp(self):
        self.train, self.held = b.build(600, 300, seed=1)

    def test_reply_has_fixed_option_order_and_valid_values(self):
        for p in self.train + self.held:
            if p["output"] == b.UNKNOWN:
                continue
            parts = p["output"].split()
            self.assertEqual(parts[:2][0], "orders")
            self.assertIn(parts[1], ("find", "count", "total"))
            opts = [w[2:] for w in parts if w.startswith("--")]
            self.assertEqual(opts, [o for o in b.ORDER if o in opts], p["output"])

    def test_request_states_every_value_of_the_reply(self):
        for p in self.train:
            words = p["output"].split()
            for key in ("--min-amount", "--max-amount", "--last-days", "--limit", "--day"):
                if key not in words:
                    continue
                v = words[words.index(key) + 1]
                stated = v in p["instruction"] or f"{int(v) / 1000:g}k" in p["instruction"] or f"{int(v):,}" in p["instruction"]
                single = key == "--limit" and v == "1"  # "the largest order": one needs no number
                spelled = int(v) in b.NUMBER_WORDS and b.NUMBER_WORDS[int(v)] in p["instruction"].lower()
                self.assertTrue(stated or single or spelled, p)

    def test_out_of_scope_requests_get_unknown_and_heldout_wording_is_new(self):
        self.assertTrue(any(p["output"] == b.UNKNOWN for p in self.train))
        held = [h for h in self.held if h["split"] == "out_of_scope"]
        self.assertTrue(held)
        trained_requests = {p["instruction"] for p in self.train}
        for h in held:
            self.assertEqual(h["output"], b.UNKNOWN)
            self.assertNotIn(h["instruction"], trained_requests)

    def test_generated_off_topic_messages_split_without_overlap(self):
        messages = [f"tell me about topic number {i}" for i in range(300)]
        train, held = b.build(2000, 600, seed=1, oos_fraction=0.5, off_topic=messages)
        side_train, side_held = b.split_off_topic(messages)
        self.assertTrue(side_held and not set(side_train) & set(side_held))
        trained = {p["instruction"] for p in train if p["output"] == b.UNKNOWN}
        heldout = {h["instruction"] for h in held if h["split"] == "out_of_scope"}
        self.assertTrue(trained & set(side_train))
        self.assertTrue(heldout & set(side_held))
        self.assertFalse(trained & set(side_held))

    def test_ids_before_after_and_short_requests_are_generated(self):
        train, _ = b.build(4000, 10, seed=2)
        outputs = [p["output"] for p in train]
        self.assertTrue(any("--id" in o for o in outputs))
        self.assertTrue(any("--before" in o for o in outputs))
        self.assertTrue(any("--after" in o for o in outputs))
        self.assertTrue(any(p["output"] == "orders find" for p in train))  # "all"
        self.assertTrue(any(len(p["instruction"].split()) <= 2 and p["output"] != b.UNKNOWN for p in train))
        for p in train:
            words = p["output"].split()
            if "--id" in words:
                self.assertEqual(len(words), 4, p)  # an id stands alone
                self.assertIn(words[3], p["instruction"])
            for key in ("--before", "--after"):
                if key in words:
                    self.assertRegex(words[words.index(key) + 1], r"^(\d{4}|[a-z]+(-\d{1,2})?(-\d{4})?)$")

    def test_search_words_are_not_off_topic_fragments(self):
        train, held = b.split_off_topic(["unpaid", "march", "what is the sea", "oct 15", "hello"])
        self.assertEqual(sorted(train + held), ["hello", "what is the sea"])

    def test_no_unfilled_template_text(self):
        for p in self.train:
            self.assertNotIn("{", p["instruction"])
            self.assertNotIn("}", p["instruction"])

    def test_unseen_combinations_never_trained(self):
        held = b.heldout_combos(seeded_rng(1), 0.1)
        for p in self.train:
            words = p["output"].split()
            status = words[words.index("--status") + 1] if "--status" in words else None
            category = words[words.index("--category") + 1] if "--category" in words else None
            self.assertNotIn((status, category), held)
        self.assertTrue(any(h["split"] == "unseen_combo" for h in self.held))

    def test_heldout_pairs_not_in_train(self):
        keys = {(p["instruction"], p["output"]) for p in self.train}
        for h in self.held:
            self.assertNotIn((h["instruction"], h["output"]), keys)

    def test_deterministic(self):
        again = b.build(600, 300, seed=1)
        self.assertEqual(again[0], self.train)


if __name__ == "__main__":
    unittest.main()
