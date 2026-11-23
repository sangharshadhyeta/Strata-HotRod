import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from serve import lowrank_maint as lm   # noqa: E402


class Result:
    def __init__(self, out="", code=0, err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


class LowrankMaint(unittest.TestCase):
    def test_coverage(self):
        c = np.zeros(512, dtype=np.uint32)
        c[1], c[2] = 60, 40
        self.assertAlmostEqual(lm.coverage({0: c}, {0: {1}}, [0])[0], 0.6)
        self.assertEqual(lm.coverage({0: np.zeros(512, dtype=np.uint32)}, {}, [0])[0], 1.0)
        self.assertEqual(lm.coverage({}, {}, [3])[3], 1.0)

    def test_settles_when_covered_or_after_three_empty_batches(self):
        self.assertEqual(lm.next_state({"calm": 0}, {0: 0.95, 1: 0.92}, 10), (True, 0))
        self.assertEqual(lm.next_state({"calm": 0}, {0: 0.5}, 4), (False, 0))
        self.assertEqual(lm.next_state({"calm": 1}, {0: 0.5}, 0), (False, 2))
        self.assertEqual(lm.next_state({"calm": 2}, {0: 0.5}, 0), (True, 3))

    def test_wakes_only_below_the_wake_share(self):
        self.assertFalse(lm.wake(True, {0: 0.85}))
        self.assertTrue(lm.wake(True, {0: 0.85, 1: 0.7}))
        self.assertFalse(lm.wake(False, {0: 0.1}))

    def test_a_batch_runs_when_idle_and_settles_on_full_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "lowrank.bin")
            open(os.path.join(d, "layer-skips.bin"), "wb").write(b"x")
            os.makedirs(os.path.join(d, "layer-capture"))
            seg = os.path.join(d, "layer-capture", "cap-1.bin")
            open(seg, "wb").write(b"x")
            os.utime(seg, (1.0, 1.0))
            seen = []
            m = lm.LowrankMaint(out, "/m/x-00001-of-00002.gguf", "/t", [0, 47], lambda: False, lambda: None, log=lambda s: None,
                                run=lambda c: seen.append(c) or Result("layer 0: 24 copies added (5 s)\nCOVERAGE 0:0.93 47:0.95 | added 24 | 10 s\n"),
                                now=lambda: 10_000.0, min_mem_gib=0.0, max_load=1e9)
            msg = m.tick()
            self.assertIn("settled", msg)
            self.assertIn("--layers", seen[0])
            self.assertTrue(m._state()["settled"])
            # settled: nothing is asked again before the hourly look
            m.last_check = 10_000.0
            self.assertIsNone(m.tick())

    def test_from_engine_args_needs_both_switches_and_layers(self):
        a = ["--native", "/m/x.gguf", "--expert-profile", "/data/expert-profile.bin"]
        env = {"STRATA_LOWRANK": "1", "STRATA_LAYER_CAPTURE": "1", "STRATA_LOWRANK_LAYERS": "0,47"}
        m = lm.from_engine_args(a, env, "/t", lambda: False, lambda: None)
        self.assertEqual(m.layers, [0, 47])
        self.assertEqual(m.out, "/data/lowrank.bin")
        self.assertIsNone(lm.from_engine_args(a, {**env, "STRATA_LOWRANK": "0"}, "/t", lambda: False, lambda: None))
        self.assertIsNone(lm.from_engine_args(a, {k: v for k, v in env.items() if k != "STRATA_LAYER_CAPTURE"}, "/t", lambda: False, lambda: None))
        self.assertIsNone(lm.from_engine_args(a, {**env, "STRATA_LOWRANK_LAYERS": ""}, "/t", lambda: False, lambda: None))


if __name__ == "__main__":
    unittest.main()
