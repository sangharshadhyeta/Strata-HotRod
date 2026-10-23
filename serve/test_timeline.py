"""The Monitor's decode timeline: the switch the engine reads, and the snapshot it writes, through /metrics and POST."""
import json
import os
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

SNAPSHOT = {"v": 1, "time": 0, "span_us": 2000,
            "layers": [[0, 0, 4, 1, 2, 0.0, 40.0, 41.0, 55.0, 700.0, 710.0, 900.0],
                       [0, 1, 4, 0, 1, 1400.0, 1440.0, 1441.0, 1455.0, 1455.0, 1460.0, 1600.0]],
            "reads": [[0, 1, 0, 50.0, 800.0]]}


class TimelineTest(unittest.TestCase):
    def setUp(self):
        for f in (server.TIMELINE_FILE, server.TIMELINE_CTL):
            if os.path.exists(f):
                os.remove(f)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello."), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for f in (server.TIMELINE_FILE, server.TIMELINE_CTL):
            if os.path.exists(f):
                os.remove(f)

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

    def test_switch_and_snapshot(self):
        self.assertEqual(self.post({"on": True}), (200, {"on": True}))
        with open(server.TIMELINE_CTL, "rb") as f:
            self.assertEqual(f.read(), b"1")
        self.assertEqual(self.metrics(), {"on": True, "data": None})          # nothing recorded yet
        with open(server.TIMELINE_FILE, "w") as f:
            json.dump(SNAPSHOT, f)
        self.assertEqual(self.metrics()["data"]["layers"], SNAPSHOT["layers"])
        old = time.time() - 60                                                 # a stale snapshot is not shown
        os.utime(server.TIMELINE_FILE, (old, old))
        self.assertIsNone(self.metrics()["data"])
        self.assertEqual(self.post({"on": False}), (200, {"on": False}))
        self.assertFalse(os.path.exists(server.TIMELINE_FILE))
        self.assertEqual(self.metrics(), {"on": False, "data": None})

    def test_rejects_bad_requests(self):
        self.assertEqual(self.post({"on": "yes"})[0], 400)
        self.assertEqual(self.post({"on": True}, ctype="text/plain")[0], 415)   # a form post from another site cannot switch it


if __name__ == "__main__":
    unittest.main()
