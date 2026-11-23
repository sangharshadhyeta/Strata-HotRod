import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve import layer_maint as lm


class Result:
    def __init__(self, code=0, out="wrote x", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


class LayerMaint(unittest.TestCase):
    def test_gguf_glob(self):
        self.assertEqual(lm.gguf_glob("/m/Qwen-UD-Q4_K_XL-00001-of-00004.gguf"), "/m/Qwen-UD-Q4_K_XL-*-of-00004.gguf")
        self.assertEqual(lm.gguf_glob("/m/model.gguf"), "/m/model.gguf")

    def gates(self, **kw):
        a = dict(now=1000.0, last_request_at=0.0, busy=False, last_run=0.0, n_segs=4, mem_gib=8.0, load1=0.1, ncpu=8)
        a.update(kw)
        return lm.should_run(**a)

    def test_runs_when_everything_is_ready(self):
        self.assertTrue(self.gates()[0])

    def test_each_gate_stops_it(self):
        self.assertFalse(self.gates(busy=True)[0])
        self.assertFalse(self.gates(last_request_at=950.0)[0])          # a request 50 s ago
        self.assertFalse(self.gates(last_run=900.0)[0])                 # ran 100 s ago
        self.assertFalse(self.gates(n_segs=3)[0])
        self.assertFalse(self.gates(mem_gib=1.0)[0])
        self.assertFalse(self.gates(load1=7.0)[0])

    def test_no_request_yet_counts_as_idle(self):
        self.assertTrue(self.gates(last_request_at=None)[0])

    def test_command_runs_at_the_lowest_priority_on_a_slice(self):
        cmd = lm.build_command("/t", ["/c/a.bin", "/c/b.bin"], "/m/x-*-of-2.gguf", "/p/layer-sens.txt", 150, 2, 4)
        self.assertEqual(cmd[:3], ["nice", "-n", "19"])
        self.assertIn("2/4", cmd)
        self.assertIn("/c/a.bin", cmd)
        self.assertEqual(cmd[cmd.index("--threads") + 1], "1")

    def test_tick_runs_the_job_consumes_segments_and_rotates_the_slice(self):
        with tempfile.TemporaryDirectory() as d:
            sens = os.path.join(d, "layer-sens.txt")
            cap = os.path.join(d, "layer-capture")
            os.makedirs(cap)
            for i in range(4):
                open(os.path.join(cap, "cap-%d.bin" % i), "wb").write(b"x")
                os.utime(os.path.join(cap, "cap-%d.bin" % i), (1.0, 1.0))
            seen = []
            m = lm.LayerMaint(sens, "/m/x-00001-of-00002.gguf", "/t", lambda: False, lambda: None, log=lambda s: None, run=lambda c: seen.append(c) or Result(),
                              now=lambda: 10_000.0, min_mem_gib=0.0, max_load=1e9)
            self.assertIsNotNone(m.tick())
            self.assertEqual(seen[0][seen[0].index("--part") + 1], "0/4")
            self.assertEqual(os.listdir(cap), [])                        # consumed
            for i in range(4):
                open(os.path.join(cap, "cap-%d.bin" % (i + 9)), "wb").write(b"x")
                os.utime(os.path.join(cap, "cap-%d.bin" % (i + 9)), (1.0, 1.0))
            m.last_run = 0.0
            self.assertIsNotNone(m.tick())
            self.assertEqual(seen[1][seen[1].index("--part") + 1], "1/4")  # the next slice

    def test_a_failing_job_keeps_the_segments_and_backs_off(self):
        with tempfile.TemporaryDirectory() as d:
            sens = os.path.join(d, "layer-sens.txt")
            cap = os.path.join(d, "layer-capture")
            os.makedirs(cap)
            for i in range(4):
                open(os.path.join(cap, "cap-%d.bin" % i), "wb").write(b"x")
                os.utime(os.path.join(cap, "cap-%d.bin" % i), (1.0, 1.0))
            now = [10_000.0]
            m = lm.LayerMaint(sens, "/m/x.gguf", "/t", lambda: False, lambda: None, log=lambda s: None, run=lambda c: Result(1, "", "boom"),
                              now=lambda: now[0], min_mem_gib=0.0, max_load=1e9)
            self.assertIn("failed", m.tick())
            self.assertEqual(len(os.listdir(cap)), 4)                    # kept
            now[0] += 300.0                                              # past the normal gap, inside the doubled one
            self.assertIsNone(m.tick())

    def test_from_engine_args(self):
        args = ["--native", "/m/x-00001-of-00002.gguf", "--expert-profile", "/data/expert-profile.bin"]
        cap = {"STRATA_LAYER_CAPTURE": "1"}
        m = lm.from_engine_args(args, cap, "/t", lambda: False, lambda: None)
        self.assertEqual(m.sens_path, "/data/layer-sens.txt")
        self.assertIsNone(lm.from_engine_args(args, {}, "/t", lambda: False, lambda: None))   # no capture asked for: nothing to refresh from
        self.assertIsNone(lm.from_engine_args(args, {"STRATA_LAYER_MAINT": "0", "STRATA_LAYER_CAPTURE": "1"}, "/t", lambda: False, lambda: None))
        self.assertIsNone(lm.from_engine_args(["--native", "/m/x.gguf"], cap, "/t", lambda: False, lambda: None))
        env = {"STRATA_LAYER_SENS": "/tmp/live/layer-sens.txt", "STRATA_LAYER_CAPTURE": "1"}
        self.assertEqual(lm.from_engine_args(["--native", "/m/x.gguf"], env, "/t", lambda: False, lambda: None).sens_path, "/tmp/live/layer-sens.txt")


if __name__ == "__main__":
    unittest.main()
