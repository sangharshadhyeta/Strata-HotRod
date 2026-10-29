"""The Monitor's "Experts in VRAM" map: the engine's residency file (serve/server.py residency_view), its environment variable,
the /metrics key, and the cleanup of the residency files of servers that no longer run."""

import json
import os
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


def good(**over):
    d = {"v": 1, "time": 0, "layers": 2, "experts": 4, "round": 3, "swaps": 5, "quality_swaps": 0,
         "rows": ["0123", "3210"], "resident": 6}
    d.update(over)
    return d


class ResidencyView(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        # the timeline's paths too: child_env turns the timeline on, which must not touch the real temp folder
        for name, value in (("RESIDENCY_FILE", str(d / "residency.json")), ("TIMELINE_FILE", str(d / "timeline.json")),
                            ("TIMELINE_CTL", str(d / "timeline.ctl"))):
            p = mock.patch.object(server, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.file = Path(server.RESIDENCY_FILE)

    def write(self, obj, age_s=0.0):
        self.file.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)
        t = time.time() - age_s
        os.utime(self.file, (t, t))

    def test_no_file_is_none(self):
        self.assertIsNone(server.residency_view())

    def test_stale_file_is_none(self):
        self.write(good(), age_s=60)
        self.assertIsNone(server.residency_view())
        self.assertEqual(server.residency_view(max_age_s=120), good())   # the age limit is the only thing that drops it

    def test_wrong_version_is_none(self):
        self.write(good(v=2))
        self.assertIsNone(server.residency_view())

    def test_rows_must_be_a_list(self):
        self.write(good(rows="0123"))
        self.assertIsNone(server.residency_view())

    def test_unreadable_file_is_none(self):
        self.write("{not json")
        self.assertIsNone(server.residency_view())

    def test_good_v1_file_is_the_dict(self):
        self.write(good())
        self.assertEqual(server.residency_view(), good())

    def test_child_env_sets_the_residency_file(self):
        env = server.child_env({"timeline": False})
        self.assertEqual(env["STRATA_RESIDENCY_FILE"], server.RESIDENCY_FILE)


class ResidencyMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(server, "RESIDENCY_FILE", str(Path(self.tmp.name) / "residency.json"))
        p.start()
        self.addCleanup(p.stop)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello."), tok,
                           ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics") as r:
            return json.loads(r.read())

    def test_metrics_carries_residency(self):
        self.assertIn("residency", self.metrics())
        self.assertIsNone(self.metrics()["residency"])
        Path(server.RESIDENCY_FILE).write_text(json.dumps(good()))
        self.assertEqual(self.metrics()["residency"], good())


class ResidencyCleanup(unittest.TestCase):
    def test_dead_servers_files_go_and_a_live_one_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            dead = os.path.join(tmp, "strata-residency-999999999.json")
            live = os.path.join(tmp, f"strata-residency-{os.getppid()}.json")   # the parent process runs
            for f in (dead, live):
                Path(f).write_text("{}")
            with mock.patch("tempfile.gettempdir", return_value=tmp):
                server._drop_stale_residency_files()
            self.assertFalse(os.path.exists(dead))
            self.assertTrue(os.path.exists(live))


if __name__ == "__main__":
    unittest.main()
