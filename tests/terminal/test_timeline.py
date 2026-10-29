"""The Monitor's decode timeline: the engine's chunks join into one rolling buffer (serve/server.py timeline_view), the switch
(POST /v1/monitor/timeline), the engine's environment, and the cleanup of the timeline's temp files."""

import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

CHUNK = {"v": 2, "time": 0, "base_ns": 1000, "now_us": 2000.0, "first": 0, "last": 0,
         "layers": [[0, 0, 4, 1, 2, 0.0, 40.0, 41.0, 55.0, 700.0, 710.0, 900.0],
                    [0, 1, 4, 0, 1, 1400.0, 1440.0, 1441.0, 1455.0, 1455.0, 1460.0, 1600.0]],
         "reads": [[0, 1, 0, 50.0, 800.0]]}


class TimelineSwitch(unittest.TestCase):
    """The HTTP side: GET /metrics carries the timeline, POST switches the engine side on and off."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        for name, value in (("TIMELINE_FILE", str(d / "timeline.json")), ("TIMELINE_CTL", str(d / "timeline.ctl"))):
            p = mock.patch.object(server, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.file, self.ctl = Path(server.TIMELINE_FILE), Path(server.TIMELINE_CTL)
        server._timeline_reset()
        self.addCleanup(server._timeline_reset)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello."), tok,
                           ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics") as r:
            return json.loads(r.read())["timeline"]

    def post(self, body, ctype="application/json"):
        req = urllib.request.Request(self.base + "/v1/monitor/timeline", data=json.dumps(body).encode(),
                                     headers={"Content-Type": ctype}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None

    def test_off_by_default_and_engine_env(self):
        self.assertEqual(self.metrics(), {"on": False, "data": None})
        env = server.child_env({})
        self.assertEqual(env["STRATA_TIMELINE_FILE"], server.TIMELINE_FILE)
        self.assertEqual(env["STRATA_TIMELINE_CTL"], server.TIMELINE_CTL)
        self.assertEqual(self.ctl.read_bytes(), b"1")          # on from the engine's start (no "timeline": false)

    def test_config_timeline_false_starts_the_engine_side_off(self):
        self.ctl.write_bytes(b"0")
        server.child_env({"timeline": False})
        self.assertEqual(self.ctl.read_bytes(), b"0")

    def test_switch_and_snapshot(self):
        self.assertEqual(self.post({"on": True}), (200, {"on": True}))
        self.assertEqual(self.ctl.read_bytes(), b"1")
        self.assertEqual(self.metrics(), {"on": True, "data": None})          # nothing recorded yet
        self.file.write_text(json.dumps(CHUNK))
        self.assertEqual(self.metrics()["data"]["layers"], CHUNK["layers"])
        with mock.patch.object(server.time, "time", return_value=time.time() + 60):   # a minute on, nothing new written:
            self.assertEqual(self.metrics()["data"]["layers"], [])                  # the chunk has aged out of the 15 s
        self.assertEqual(self.post({"on": False}), (200, {"on": False}))
        self.assertFalse(self.file.exists())
        self.assertEqual(self.metrics(), {"on": False, "data": None})

    def test_rejects_bad_requests(self):
        self.assertEqual(self.post({"on": "yes"})[0], 400)
        self.assertEqual(self.post({"on": True}, ctype="text/plain")[0], 415)   # a form post from another site cannot switch it


class TimelineBuffer(unittest.TestCase):
    """The Monitor's decode timeline (timeline_view): the engine's chunks join into one rolling buffer of the last 15 s."""

    def setUp(self):
        self.srv = server
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.file, self.ctl = d / "timeline.json", d / "timeline.ctl"
        self.ctl.write_bytes(b"1")
        for name, value in (("TIMELINE_FILE", str(self.file)), ("TIMELINE_CTL", str(self.ctl))):
            p = mock.patch.object(server, name, value)
            p.start()
            self.addCleanup(p.stop)
        server._timeline_reset()
        self.addCleanup(server._timeline_reset)

    @staticmethod
    def row(w, k):
        t = w * 100_000.0 + k * 1_000.0               # a window is 100 ms; two layers a window
        return [w, k, 1, 0, 0, t, t + 10, t + 20, t + 30, t + 40, t + 50, t + 60]

    def write(self, first, last, now_us=None, base=1000, gap=None, reads=(), v=2):
        c = {"v": v, "time": 0, "base_ns": base, "now_us": now_us if now_us is not None else last * 100_000.0 + 100_000.0,
             "first": first, "last": last, "layers": [self.row(w, k) for w in range(first, last + 1) for k in range(2)],
             "reads": [list(r) for r in reads]}
        if gap:
            c["gap"] = gap
        self.file.write_text(json.dumps(c))

    def windows(self, data):
        return sorted({r[0] for r in data["layers"]})

    def test_new_chunk_is_appended_once(self):
        self.write(0, 2)
        data = self.srv.timeline_view()["data"]
        self.assertEqual(data["v"], 2)
        self.assertEqual(self.windows(data), [0, 1, 2])
        self.assertEqual(len(data["layers"]), 6)
        self.assertEqual((data["first"], data["last"]), (0, 2))
        again = self.srv.timeline_view()["data"]        # the same file read again adds nothing
        self.assertEqual(len(again["layers"]), 6)

    def test_overlapping_chunk_adds_only_the_new_windows(self):
        self.write(0, 2)
        self.srv.timeline_view()
        self.write(1, 3)                                 # the engine's next write repeats window 1 and 2
        data = self.srv.timeline_view()["data"]
        self.assertEqual(self.windows(data), [0, 1, 2, 3])
        self.assertEqual(len(data["layers"]), 8)
        self.assertEqual(data["last"], 3)

    def test_stale_and_duplicate_chunks_are_ignored(self):
        self.write(0, 3)
        self.srv.timeline_view()
        self.write(1, 2)                                 # older than the buffer's last window
        self.assertEqual(len(self.srv.timeline_view()["data"]["layers"]), 8)
        self.write(0, 3)                                 # the same chunk again
        self.assertEqual(len(self.srv.timeline_view()["data"]["layers"]), 8)

    def test_a_chunk_of_another_version_is_ignored(self):
        self.write(0, 2, v=1)
        self.assertIsNone(self.srv.timeline_view()["data"])

    def test_reads_are_joined_once(self):
        reads = [[0, 3, 1, 500.0, 900.0], [1, 3, 0, 600.0, 700.0]]
        self.write(0, 2, reads=reads)
        self.srv.timeline_view()
        self.write(1, 3, reads=reads)                    # the overlap carries the same reads
        data = self.srv.timeline_view()["data"]
        self.assertEqual(len(data["reads"]), 2)
        self.write(4, 4, reads=[[0, 4, 0, 4_000_000.0, 4_000_500.0]])
        self.assertEqual(len(self.srv.timeline_view()["data"]["reads"]), 3)

    def test_gap_is_marked_at_the_first_window_of_its_chunk(self):
        self.write(0, 1)
        self.srv.timeline_view()
        self.write(5, 6, gap=3)
        data = self.srv.timeline_view()["data"]
        self.assertEqual(data["gaps"], [[500_000.0, 3]])

    def test_fifteen_seconds_are_kept(self):
        self.write(0, 2, reads=[[0, 0, 0, 100.0, 200.0]])
        self.srv.timeline_view()
        self.write(200, 201, now_us=20_100_000.0)        # 20 s on: the first chunk is older than 15 s
        data = self.srv.timeline_view()["data"]
        self.assertEqual(self.windows(data), [200, 201])
        self.assertEqual(data["reads"], [])
        self.assertEqual(len(self.srv._TL["layers"]), 4)

    def test_rows_are_bounded(self):
        self.write(0, 2)
        self.srv.timeline_view()
        with mock.patch.object(self.srv, "TIMELINE_MAX_ROWS", 4):
            self.write(3, 9)
            data = self.srv.timeline_view()["data"]
        self.assertEqual(len(data["layers"]), 4)
        self.assertEqual(data["layers"][-1][0], 9)

    def test_a_new_engine_run_starts_the_buffer_over(self):
        self.write(0, 2, base=1000)
        self.srv.timeline_view()
        self.write(0, 1, base=2000)
        data = self.srv.timeline_view()["data"]
        self.assertEqual(data["base_ns"], 2000)
        self.assertEqual(self.windows(data), [0, 1])
        self.assertEqual(len(data["layers"]), 4)

    def test_switch_off_clears_the_buffer(self):
        self.write(0, 2)
        self.assertIsNotNone(self.srv.timeline_view()["data"])
        self.srv.set_timeline(False)
        self.assertEqual(self.srv.timeline_view(), {"on": False, "data": None})
        self.assertIsNone(self.srv._TL["last"])
        self.assertFalse(self.file.exists())

    def test_the_engine_side_off_returns_nothing(self):
        self.write(0, 2)
        self.ctl.write_bytes(b"0")
        self.assertEqual(self.srv.timeline_view(), {"on": False, "data": None})

    def test_incremental_poll_returns_only_the_rows_after_the_page_s_newest(self):
        self.write(0, 2)
        self.srv.timeline_view()
        self.write(1, 3)
        after = self.row(2, 1)[11]                        # the page holds window 2 through its last layer
        data = self.srv.timeline_view(after, 1000)["data"]
        self.assertFalse(data["reset"])
        self.assertEqual(self.windows(data), [3])
        self.assertEqual(len(data["layers"]), 2)
        self.assertEqual(data["last"], 3)

    def test_incremental_poll_returns_the_reads_after_it_too(self):
        reads = [[0, 3, 1, 500.0, 900.0], [1, 3, 0, 600.0, 700.0]]
        self.write(0, 2, reads=reads)
        self.srv.timeline_view()
        self.write(3, 3, reads=[[0, 3, 0, 300_000.0, 300_500.0]])
        data = self.srv.timeline_view(self.row(2, 1)[11], 1000)["data"]
        self.assertEqual(data["reads"], [[0, 3, 0, 300_000.0, 300_500.0]])

    def test_no_after_returns_the_last_five_seconds_and_resets(self):
        self.write(0, 2)
        self.srv.timeline_view()
        self.write(0, 100)                                # now 10.1 s: the last 5 s are windows 51 to 100
        with mock.patch.object(self.srv.time, "time", return_value=os.stat(self.file).st_mtime):   # (no file age)
            data = self.srv.timeline_view()["data"]
        self.assertTrue(data["reset"])
        self.assertEqual(min(self.windows(data)), 51)
        self.assertEqual(len(data["layers"]), 100)
        self.assertEqual(data["first"], 0)               # (the buffer's own first window: the page is told what it holds)

    def test_a_new_run_resets_even_with_an_after(self):
        self.write(0, 2, base=1000)
        self.srv.timeline_view()
        self.write(0, 1, base=2000)
        data = self.srv.timeline_view(self.row(2, 1)[11], 1000)["data"]
        self.assertTrue(data["reset"])
        self.assertEqual(data["base_ns"], 2000)
        self.assertEqual(self.windows(data), [0, 1])

    def test_query_parameters_are_read(self):
        self.assertEqual(self.srv.timeline_query("/metrics?tl_after=123&tl_base=456"), (123, 456))
        self.assertEqual(self.srv.timeline_query("/metrics"), (None, None))
        self.assertEqual(self.srv.timeline_query("/metrics?tl_after=abc"), (None, None))


class StaleTimelineFiles(unittest.TestCase):
    """At start, the timeline files of servers that no longer run go; a live server's are kept, and so is any other name."""

    def test_stale_files_go_and_live_ones_stay(self):
        live = os.getpid() + 1
        dead = live + 1
        with tempfile.TemporaryDirectory() as tmp:
            names = [f"strata-timeline-{dead}.json", f"strata-timeline-{dead}.ctl", f"strata-timeline-{live}.ctl",
                     f"strata-timeline-{os.getpid()}.ctl", f"strata-timeline-{dead}.json.tmp", "strata-timeline-x.ctl",
                     "strata-residency-1.json"]
            for n in names:
                Path(tmp, n).write_bytes(b"1")
            with mock.patch.object(server.tempfile, "gettempdir", return_value=tmp), \
                    mock.patch.object(server, "_pid_alive", side_effect=lambda p: p == live):
                server._drop_stale_timeline_files()
            left = sorted(os.listdir(tmp))
        self.assertEqual(left, sorted([f"strata-timeline-{live}.ctl", "strata-timeline-x.ctl", "strata-residency-1.json"]))


if __name__ == "__main__":
    unittest.main()
