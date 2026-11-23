#!/usr/bin/env python3
"""Tests of quality_audit's arithmetic on small hand-made STRATA_LOGPOS rows (no server needed)."""
import math, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality_audit as qa


def row(pos, target, lp_target, top, lp_top, cands):
    return "\t".join([str(pos), str(target), "%.9f" % lp_target, str(top), "%.9f" % lp_top, str(int(top == target)), "nan", "nan"] + ["%d:%.6f" % (i, v) for i, v in cands])


def dist(probs):
    return [(i, math.log(p)) for i, p in probs]


class AuditArithmetic(unittest.TestCase):
    def setUp(self):
        a = dist([(5, 0.6), (7, 0.3), (9, 0.05)])           # 0.05 left in the rest bucket
        self.ref = qa.parse_logpos("\n".join([row(0, 5, math.log(0.6), 5, math.log(0.6), a), row(1, 7, math.log(0.3), 5, math.log(0.6), a)]))

    def test_identical_runs_have_no_difference(self):
        n, dn, t1, kl = qa.compare(self.ref, self.ref)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(dn, 0.0)
        self.assertEqual(t1, 1.0)
        self.assertAlmostEqual(kl, 0.0, places=9)

    def test_a_worse_run_costs_nll_kl_and_flips_the_top_token(self):
        b = dist([(7, 0.5), (5, 0.3), (9, 0.1)])             # the setting now prefers token 7
        got = qa.parse_logpos("\n".join([row(0, 5, math.log(0.3), 7, math.log(0.5), b), row(1, 7, math.log(0.5), 7, math.log(0.5), b)]))
        n, dn, t1, kl = qa.compare(self.ref, got)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(dn, ((-math.log(0.3) + math.log(0.6)) + (-math.log(0.5) + math.log(0.3))) / 2, places=9)
        self.assertEqual(t1, 0.0)
        self.assertGreater(kl, 0.0)

    def test_only_shared_positions_count(self):
        part = {0: self.ref[0]}
        n, dn, t1, kl = qa.compare(self.ref, part)
        self.assertEqual(n, 1)

    def test_configs_parse(self):
        cfg = qa.parse_configs("a:tail_weight=0,tail_skip=0;b:tail_weight=0.05,tail_skip=7")
        self.assertEqual(cfg, [("a", {"tail_weight": 0.0, "tail_skip": 0}), ("b", {"tail_weight": 0.05, "tail_skip": 7})])
        self.assertEqual(len(qa.parse_configs(qa.DEFAULT_CONFIGS)), 9)

    def test_the_audit_text_fits_the_short_read(self):
        import json
        msgs = json.load(open(os.path.join(os.path.dirname(qa.__file__), "audit_text.json"), encoding="utf-8"))
        self.assertGreaterEqual(len(msgs), 8)
        self.assertTrue(all(len(m["content"]) < 2400 for m in msgs))   # about 640 tokens at 3.3 characters a token, with room


if __name__ == "__main__":
    unittest.main()
