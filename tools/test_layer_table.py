import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_table as lt


class LayerTable(unittest.TestCase):
    def test_roundtrip(self):
        text = lt.format_table([0.1, 0.2, 0.4], 3, "x")
        self.assertEqual(lt.parse_table(text), [0.1, 0.2, 0.4])
        self.assertEqual(lt.parse_runs(text), 3)

    def test_first_run_takes_the_measurement(self):
        self.assertEqual(lt.merge_table(None, [0.1, 0.3], 0.3), [0.1, 0.3])

    def test_blend(self):
        out = lt.merge_table([0.2, 0.4], [0.4, 0.2], 0.25)
        self.assertAlmostEqual(out[0], 0.75 * 0.2 + 0.25 * 0.4)
        self.assertAlmostEqual(out[1], 0.75 * 0.4 + 0.25 * 0.2)

    def test_unmeasured_layers_keep_their_value(self):
        out = lt.merge_table([0.2, 0.4], [None, 0.2], 0.5)
        self.assertEqual(out[0], 0.2)
        self.assertAlmostEqual(out[1], 0.3)

    def test_other_size_table_is_replaced(self):
        self.assertEqual(lt.merge_table([0.2], [0.4, 0.6], 0.3), [0.4, 0.6])

    def test_missing_layers_are_filled_with_the_mean(self):
        vals = lt.parse_table(lt.format_table([0.2, None, 0.4], 1, "x"))
        self.assertAlmostEqual(vals[1], 0.3, places=4)


    def test_placeholders_are_not_blended(self):
        text = lt.format_table([0.2, None, 0.4], 1, "x")
        self.assertEqual(lt.parse_measured(text), {0, 2})
        old = lt.parse_table(text)
        out = lt.merge_table(old, [None, 0.6, None], 0.3, lt.parse_measured(text))
        self.assertAlmostEqual(out[1], 0.6)            # the placeholder (the mean 0.3) did not count
        self.assertEqual(out[0], 0.2)


    def test_pairs_roundtrip(self):
        text = lt.format_table([0.2, None, 0.4], 2, "x", [0.03, None, 0.05])
        sens, stat = lt.parse_pairs(text)
        self.assertEqual(len(sens), 3)
        self.assertEqual(len(stat), 3)
        self.assertAlmostEqual(stat[0], 0.03)
        self.assertEqual(lt.parse_table(text)[0], 0.2)

    def test_one_number_table_has_no_statistics(self):
        self.assertEqual(lt.parse_pairs(lt.format_table([0.2, 0.4], 1, "x"))[1], [])


if __name__ == "__main__":
    unittest.main()
