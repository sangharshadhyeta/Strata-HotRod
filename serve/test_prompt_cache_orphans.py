"""The disk prompt cache removes state files a save did not finish, and never one another server may still be writing."""
import os
import tempfile
import time
import unittest

from serve.prompt_cache import PromptCache


class Died(RuntimeError):
    pass


class DyingEngine:
    gen = 1

    def session_file(self, action, path):
        with open(path, "wb") as f:      # the engine wrote part of the state, then died
            f.write(b"x" * 1024)
        raise Died("the engine died")


class Orphans(unittest.TestCase):
    def test_scan_removes_old_orphans_only(self):
        with tempfile.TemporaryDirectory() as d:
            old = time.time() - 7200
            for name in ("aaa.bin", "aaa.bin.conv", "bbb.json.tmp"):
                with open(os.path.join(d, name), "wb") as f:
                    f.write(b"x")
                os.utime(os.path.join(d, name), (old, old))
            with open(os.path.join(d, "ccc.bin"), "wb") as f:   # being written by another server right now
                f.write(b"x")
            PromptCache(d, "fp")
            self.assertEqual(sorted(os.listdir(d)), ["ccc.bin"])

    def test_a_save_that_dies_leaves_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            pc = PromptCache(d, "fp", min_read=1)
            with self.assertRaises(Died):
                pc.after(DyingEngine(), [1, 2, 3], [4], prompt_read=100)
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()
