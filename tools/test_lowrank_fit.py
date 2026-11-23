import os, sys, struct, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowrank_fit as lf


class LowrankFit(unittest.TestCase):
    def test_q4_roundtrip_is_close(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((16, 70)).astype(np.float32)
        sc, pk = lf.quantize_q4(M)
        R = lf.dequantize_q4(sc, pk, 70)
        self.assertEqual(R.shape, M.shape)
        self.assertLess(float(np.abs(R - M).max()), float(np.abs(M).max()) / 7.0 * 0.6 + 1e-3)   # within half a step of the group's scale

    def test_record_size_matches_the_packed_bytes(self):
        rng = np.random.default_rng(1)
        r = 8
        f = {"gate": (rng.standard_normal((640, r)), rng.standard_normal((r, 2560))), "up": (rng.standard_normal((640, r)), rng.standard_normal((r, 2560))),
             "down": (rng.standard_normal((2560, r)), rng.standard_normal((r, 640)))}
        self.assertEqual(len(lf.pack_record(3, 77, r, f)), lf.record_size(r))

    def test_read_records_skips_through_the_file(self):
        rng = np.random.default_rng(2)
        r = 4
        f = {"gate": (rng.standard_normal((640, r)), rng.standard_normal((r, 2560))), "up": (rng.standard_normal((640, r)), rng.standard_normal((r, 2560))),
             "down": (rng.standard_normal((2560, r)), rng.standard_normal((r, 640)))}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "lowrank.bin")
            with open(p, "wb") as fh:
                fh.write(lf.MAGIC + struct.pack("<I", 1))
                fh.write(lf.pack_record(0, 5, r, f))
                fh.write(lf.pack_record(0, 9, r, f))
                fh.write(lf.pack_record(47, 2, r, f))
            self.assertEqual(lf.read_records(p), [(0, 5, r), (0, 9, r), (47, 2, r)])
            self.assertEqual(lf.read_records(os.path.join(d, "none.bin")), [])

    def test_pick_most_skipped_first_and_stop_at_coverage(self):
        c = np.zeros(512, dtype=np.uint32)
        c[10], c[20], c[30], c[40] = 50, 30, 15, 5
        chosen, cov = lf.pick_experts(c, set(), 10, 0.9)
        self.assertEqual(chosen, [10, 20, 30])        # 95 of 100 after three: stops
        self.assertAlmostEqual(cov, 0.95)
        chosen, cov = lf.pick_experts(c, {10, 20, 30}, 10, 0.99)
        self.assertEqual(chosen, [40])
        chosen, cov = lf.pick_experts(c, set(), 2, 1.0)
        self.assertEqual(chosen, [10, 20])             # the batch limit

    def test_nothing_skipped_nothing_to_fit(self):
        self.assertEqual(lf.pick_experts(np.zeros(512, dtype=np.uint32), set(), 5, 0.9), ([], 1.0))

    def test_read_skips(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "layer-skips.bin")
            data = np.arange(2 * 512, dtype=np.uint32).reshape(2, 512)
            with open(p, "wb") as fh:
                fh.write(struct.pack("<III", 0x314B534C, 2, 512))
                fh.write(data.tobytes())
            s = lf.read_skips(p)
            self.assertEqual(int(s[1][3]), 512 + 3)


if __name__ == "__main__":
    unittest.main()
