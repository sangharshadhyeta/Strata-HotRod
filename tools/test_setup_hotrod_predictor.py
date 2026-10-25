"""Tests for the learned look-ahead's download (setup.fetch_hotrod_predictor): fetched for Qwen IQ3_S only, SHA-256 checked, a
PC's own learned file kept, a failed or wrong download leaving setup running.  Every outside effect is local (file:// sources).

    python -m unittest tools.test_setup_hotrod_predictor
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class FetchHotrodPredictor(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-hotrod-pred-"))
        self.src = self.tmp / "src.bin"
        self.src.write_bytes(b"learned look-ahead state" * 100)
        self.sha = hashlib.sha256(self.src.read_bytes()).hexdigest()
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.dst = self.root / "data" / setup.HOTROD_PREDICTOR

    def fetch(self, family="qwen", model="IQ3_S", sha=None, url=None):
        with mock.patch.object(setup, "ROOT", self.root), \
             mock.patch.object(setup, "HOTROD_PREDICTOR_URL", url or "file://" + str(self.src)), \
             mock.patch.object(setup, "HOTROD_PREDICTOR_SHA", sha or self.sha), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            setup.fetch_hotrod_predictor(family, model)
        return out.getvalue()

    def test_downloads_and_checks(self):
        self.fetch()
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())
        self.assertFalse(list(self.dst.parent.glob("*.download*")), "no download leftovers")

    def test_other_model_or_family_gets_nothing(self):
        self.fetch(model="Q2_0")
        self.fetch(family="coder")
        self.assertFalse(self.dst.exists())

    def test_wrong_sha_is_refused_and_setup_goes_on(self):
        out = self.fetch(sha="0" * 64)
        self.assertFalse(self.dst.exists())
        self.assertIn("wrong SHA-256", out)

    def test_missing_source_leaves_it_off(self):
        out = self.fetch(url="file://" + str(self.tmp / "missing.bin"))
        self.assertFalse(self.dst.exists())
        self.assertIn("stays off", out)

    def test_own_learned_file_is_kept(self):
        self.dst.parent.mkdir(parents=True)
        self.dst.write_bytes(b"this PC learned more")
        self.fetch()
        self.assertEqual(self.dst.read_bytes(), b"this PC learned more")


if __name__ == "__main__":
    unittest.main()
