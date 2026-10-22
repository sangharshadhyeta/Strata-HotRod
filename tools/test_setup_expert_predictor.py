"""Tests for --expert-predictor (opt-in): off by default, written as STRATA_LA_MODEL when asked for, kept by a setup run again
without the flag, refused for a model it was not trained on, and the one-time fetch (download, SHA-256, conversion, cleanup).
Every outside effect is mocked (tools/test_setup_golden.py's install()).

    python -m unittest tools.test_setup_expert_predictor
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
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402

FAKE = Path(tempfile.gettempdir()) / "strata-test-la_model.bin"


def posix(p) -> str:
    """install() normalizes the config's paths to forward slashes."""
    return str(p).replace("\\", "/")


class Choice(unittest.TestCase):
    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]
        self.fetch = mock.Mock(return_value=FAKE)

    def run_setup(self, family, model, *extra, configs=()):
        return install(self.ram, self.found, ["--family", family, "--model", model, "--no-start", *extra], configs=configs,
                       extra=[mock.patch.object(setup, "fetch_expert_predictor", self.fetch)])

    def test_off_by_default(self):
        code, out, cfg, _ = self.run_setup("qwen", "Q2_0")
        self.assertEqual(code, 0, out)
        self.assertNotIn("STRATA_LA_MODEL", cfg.get("env") or {})
        self.assertIn("expert predictor: off", out)
        self.fetch.assert_not_called()

    def test_on_writes_the_env_and_a_rerun_keeps_it(self):
        code, out, cfg, _ = self.run_setup("qwen", "Q2_0", "--expert-predictor", "on")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["env"]["STRATA_LA_MODEL"], posix(FAKE))
        self.assertIn("expert predictor: ON", out)
        old = {**cfg, "env": {**cfg["env"], "STRATA_LA_MODEL": str(setup.LA_MODEL)}}   # what a real install wrote
        code, out, cfg2, _ = self.run_setup("qwen", "Q2_0", configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg2["env"]["STRATA_LA_MODEL"], posix(FAKE))                  # kept: the fetch ran again (it is a no-op once made)
        code, out, cfg3, _ = self.run_setup("qwen", "Q2_0", "--expert-predictor", "off", configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        self.assertNotIn("STRATA_LA_MODEL", cfg3.get("env") or {})                    # off removes it (a setup-written key)

    def test_a_file_of_ones_own(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "mine.bin"
            f.write_bytes(b"LAM1")
            code, out, cfg, _ = self.run_setup("qwen", "Q2_0", "--expert-predictor", str(f))
            self.assertEqual(code, 0, out)
            self.assertEqual(cfg["env"]["STRATA_LA_MODEL"], posix(f.resolve()))
            self.fetch.assert_not_called()
            code, out, _, _ = self.run_setup("qwen", "Q2_0", "--expert-predictor", str(Path(d) / "missing.bin"))
            self.assertEqual(code, 1)
            self.assertIn("the expert predictor file is missing", out)

    def test_not_for_another_model(self):
        code, out, cfg, _ = self.run_setup("swift", "IQ3_S", "--expert-predictor", "on")
        self.assertEqual(code, 0, out)
        self.assertNotIn("STRATA_LA_MODEL", cfg.get("env") or {})
        self.assertIn("trained on the original Qwen3.8-Flash-Next, not Swift 1.5: left off", out)
        self.fetch.assert_not_called()


class Fetch(unittest.TestCase):
    def fetch(self, tar_bytes, sha):
        with tempfile.TemporaryDirectory() as d:
            la = Path(d) / "data" / "expert-predictor" / "la_model.bin"
            calls = []

            def fake_download(url, dst, what=None):
                calls.append(url)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tar_bytes)
                setup.mark(dst)

            def fake_run(cmd, **kw):
                calls.append([str(c) for c in cmd])
                Path(cmd[-1]).write_bytes(b"LAM1")

            out = io.StringIO()
            with mock.patch.object(setup, "LA_MODEL", la), mock.patch.object(setup, "LA_MODEL_SHA", sha), \
                    mock.patch.object(setup, "download", fake_download), mock.patch.object(setup, "run", fake_run), \
                    contextlib.redirect_stdout(out):
                try:
                    got = setup.fetch_expert_predictor({})
                    code = 0
                except SystemExit as e:
                    got, code = None, e.code
                again = setup.fetch_expert_predictor({}) if code == 0 else None
            return dict(code=code, got=got, again=again, calls=calls, out=out.getvalue(), la=la,
                        tar_left=any((la.parent / n).exists() for n in ("197_models_d48_linear.tar", "197_models_d48_linear.tar.done")),
                        made=la.is_file(), marked=la.with_name(la.name + ".done").exists())

    def test_download_check_convert_and_clean_up(self):
        data = b"a tar"
        r = self.fetch(data, hashlib.sha256(data).hexdigest())
        self.assertEqual(r["code"], 0, r["out"])
        self.assertEqual(r["got"], r["la"])
        self.assertTrue(r["made"] and r["marked"])
        self.assertFalse(r["tar_left"])                                              # the 774 MB tar does not stay
        self.assertIn("/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces/resolve/" + setup.LA_MODEL_REV, r["calls"][0])
        self.assertTrue(r["calls"][1][1].endswith("export_la_model.py"))
        self.assertEqual(len(r["calls"]), 2)                                          # the second call found it made
        self.assertIn("expert predictor already prepared", r["out"])

    def test_wrong_sha_stops_and_removes_the_download(self):
        r = self.fetch(b"something else", "0" * 64)
        self.assertEqual(r["code"], 1)
        self.assertIn("wrong SHA-256", r["out"])
        self.assertFalse(r["tar_left"] or r["made"])
        self.assertEqual(len(r["calls"]), 1)                                          # nothing converted


class Saved(unittest.TestCase):
    def test_choice_from_flag_and_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "strata-x.json"
            self.assertEqual(setup.expert_predictor_choice(None, cfg), "off")          # no config yet
            cfg.write_text('{"env": {"STRATA_LA_MODEL": "%s"}}' % str(setup.LA_MODEL).replace("\\", "\\\\"))
            self.assertEqual(setup.expert_predictor_choice(None, cfg), "on")
            cfg.write_text('{"env": {"STRATA_LA_MODEL": "/x/own.bin"}}')
            self.assertEqual(setup.expert_predictor_choice(None, cfg), "/x/own.bin")
            self.assertEqual(setup.expert_predictor_choice("OFF", cfg), "off")
            self.assertEqual(setup.expert_predictor_choice("yes", cfg), "on")
            cfg.write_text("not json")
            self.assertEqual(setup.expert_predictor_choice(None, cfg), "off")


if __name__ == "__main__":
    unittest.main()
