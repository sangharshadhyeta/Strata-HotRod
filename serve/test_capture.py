"""serve/test_capture.py - the capture record (docs/SPEC-capture.md, sections A and B, the ring, the reader, the engine hook).

Fakes and temporary folders only.  Run with the other suites:
    PYTHONPATH=serve python -m unittest serve.test_capture
"""
import array
import contextlib
import json
import os
import shutil
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import capture as cap
from serve import capture_read
from serve.capture_format import encode_header, encode_record
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
TURN = 7                                   # a stand-in turn token (the real one is StrataEngine.TURN_ID)
CONV_A = [TURN, 1, 2, TURN, 3, 4, TURN, 5]  # system, user, assistant: the conversation key is these eight ids
CONV_B = [TURN, 9, 9, TURN, 8, 8, TURN, 6]  # another conversation


def ids(xs):
    return array.array("i", xs).tobytes()


class TempFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="capture-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dir = os.path.join(self.tmp, "capture")


class RoundTrip(TempFolder):
    def test_a_b_and_engine_kinds_round_trip(self):
        c = cap.Capture(self.dir, "fp-1", model="m1", turn_id=TURN)
        self.assertTrue(c.request(CONV_A + [40, 41], [50, 51, 52], "stop", {"reasoning_tokens": 2, "completion_tokens": 3}))
        self.assertTrue(c.add_preference("req-x", True))
        self.assertTrue(c.add_engine_record("C", {"layer": 4, "count": 3},
                                            {"sum": ("f4", array.array("f", [1.5, -2.0])), "pos": ("i4", array.array("i", [7]))}))
        self.assertTrue(c.add_engine_record("D", {"token": 12}, {"top_ids": ("i4", array.array("i", [1, 2, 3]))}))
        self.assertTrue(c.add_engine_record("E", {"expert": 5}, {"y": ("f2", bytes(8))}))
        c.close()
        recs = list(capture_read.iter_records(self.dir))
        kinds = [r["kind"] for r in recs]
        self.assertEqual(kinds, ["A", "B", "C", "D", "E"])
        a = recs[0]
        self.assertEqual(a["meta"]["finish"], "stop")
        self.assertEqual(a["meta"]["completion_tokens"], 3)
        self.assertEqual(a["meta"]["conv_len"], 7)          # up to and including the third turn token (index 6)
        self.assertEqual(a["arrays"]["prompt"].tolist(), CONV_A + [40, 41])
        self.assertEqual(a["arrays"]["answer"].tolist(), [50, 51, 52])
        self.assertEqual(recs[1]["meta"], {"type": "explicit", "req": "req-x", "chosen": True, "t": recs[1]["meta"]["t"]})
        self.assertEqual(recs[2]["meta"]["layer"], 4)
        self.assertEqual(recs[2]["arrays"]["sum"].tolist(), [1.5, -2.0])
        self.assertEqual(recs[3]["arrays"]["top_ids"].tolist(), [1, 2, 3])
        self.assertEqual(recs[4]["arrays"]["y"], bytes(8))

    def test_header_and_owner_only_modes(self):
        c = cap.Capture(self.dir, "fp-9", model="m9", turn_id=TURN)
        c.request(CONV_A, [1], "stop", {})
        c.close()
        paths = capture_read.segment_paths(self.dir)
        self.assertEqual(len(paths), 1)
        h = capture_read.read_segment_header(paths[0])
        self.assertEqual((h["format"], h["fingerprint"], h["model"], h["seq"]), (1, "fp-9", "m9", 1))
        self.assertEqual(os.stat(self.dir).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(paths[0]).st_mode & 0o777, 0o600)

    def test_fingerprint_filter_and_bad_header(self):
        c = cap.Capture(self.dir, "fp-A", turn_id=TURN)
        c.request(CONV_A, [1], "stop", {})
        c.close()
        self.assertEqual(len(list(capture_read.iter_records(self.dir, fingerprint="fp-A"))), 1)
        self.assertEqual(list(capture_read.iter_records(self.dir, fingerprint="other")), [])
        with open(os.path.join(self.dir, "seg-00000099.cap"), "wb") as f:
            f.write(b"NOTCAPTURE" + bytes(40))
        with self.assertRaises(ValueError):
            list(capture_read.iter_records(self.dir))
        self.assertEqual(len(list(capture_read.iter_records(self.dir, skip_bad=True))), 1)

    def test_torn_tail_is_not_read(self):
        os.makedirs(self.dir)
        path = os.path.join(self.dir, "seg-00000001.cap")
        with open(path, "wb") as f:
            f.write(encode_header("fp", "m", 1, time.time()))
            f.write(encode_record("B", {"type": "explicit", "req": "one", "chosen": True}))
            f.write(encode_record("B", {"type": "explicit", "req": "two", "chosen": False})[:-5])
        got = [r["meta"]["req"] for r in capture_read.iter_records(self.dir)]
        self.assertEqual(got, ["one"])

    def test_engine_hook_refuses_other_kinds(self):
        c = cap.Capture(self.dir, "fp", start=False)
        with self.assertRaises(ValueError):
            c.add_engine_record("A", {}, {})
        c.close()


class Ring(TempFolder):
    def test_size_cap_drops_the_oldest_segments(self):
        limit = 200 << 10
        c = cap.Capture(self.dir, "fp", turn_id=TURN, cap_bytes=limit)
        prompt = CONV_A + list(range(100, 2600))           # about 10 KB of ids per record
        for _ in range(60):
            c.request(prompt, [1, 2], "stop", {})
        c.close()
        total = sum(os.path.getsize(p) for p in capture_read.segment_paths(self.dir))
        self.assertLessEqual(total, limit)
        seqs = [capture_read.read_segment_header(p)["seq"] for p in capture_read.segment_paths(self.dir)]
        self.assertGreater(seqs[0], 1)                      # the first segments went
        kept = [r["meta"]["req"] for r in capture_read.iter_records(self.dir, kinds={"A"})]
        self.assertTrue(kept and kept[-1].endswith("-00000060"))

    def test_limit_is_the_smaller_of_gib_and_a_sixteenth_of_free(self):
        usage = mock.Mock(free=32 << 30)
        with mock.patch.object(cap.shutil, "disk_usage", return_value=usage):
            self.assertEqual(cap.Capture(self.dir, "fp", gib=4.0, start=False)._cap(), 2 << 30)   # free/16 = 2 GiB
        usage = mock.Mock(free=128 << 30)
        with mock.patch.object(cap.shutil, "disk_usage", return_value=usage):
            self.assertEqual(cap.Capture(self.dir, "fp", gib=4.0, start=False)._cap(), 4 << 30)   # gib is the smaller
            self.assertEqual(cap.Capture(self.dir, "fp", gib=0.25, start=False)._cap(), (1 << 30) // 4)

    def test_segments_unwritten_past_the_age_go_at_open(self):
        os.makedirs(self.dir)
        old = os.path.join(self.dir, "seg-00000001.cap")
        new = os.path.join(self.dir, "seg-00000002.cap")
        for p in (old, new):
            with open(p, "wb") as f:
                f.write(encode_header("fp", "m", 1, time.time()))
        past = time.time() - 15 * 86400
        os.utime(old, (past, past))
        c = cap.Capture(self.dir, "fp", max_age_d=14, start=False)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(new))
        self.assertEqual(c._seq, 3)
        c.close()

    def test_settings_are_off_unless_asked_and_validated(self):
        target = os.path.join(self.tmp, "should-not-exist")
        self.assertIsNone(cap.settings({"capture_path": target}, {}))
        self.assertFalse(os.path.exists(target))
        self.assertIsNone(cap.settings({"capture": False}, {"STRATA_CAPTURE": "0"}))
        on = cap.settings({"capture_path": target}, {"STRATA_CAPTURE": "1"})
        self.assertEqual((on["gib"], on["max_age_d"]), (4.0, 14.0))
        self.assertTrue(os.path.isdir(target))
        self.assertIsNotNone(cap.settings({"capture": True, "capture_path": self.dir}, {}))
        for bad in ({"capture": "yes"}, {"capture": True, "capture_gib": 0}, {"capture": True, "capture_max_age_d": -1},
                    {"capture": True, "capture_path": "x\x01y"}):
            with self.assertRaises(ValueError):
                cap.settings(bad, {})


class Regenerate(TempFolder):
    def test_same_prompt_is_replaced_and_the_continued_answer_is_kept(self):
        c = cap.Capture(self.dir, "fp", turn_id=TURN)
        c.request(CONV_A + [40], [50, 51, 52, 53], "stop", {})      # first answer
        c.request(CONV_A + [40], [60, 61, 62, 63], "stop", {})      # a regenerate of the same prompt
        c.request(CONV_A + [40, 60, 61, 62, 63, 41], [70], "stop", {})  # the conversation continues with the second answer
        c.close()
        pairs = capture_read.preference_pairs(self.dir)
        self.assertEqual(len(pairs), 1)
        recs = {r["meta"]["req"]: r for r in capture_read.iter_records(self.dir, kinds={"A"})}
        first, second = sorted(recs)[0], sorted(recs)[1]
        self.assertEqual(pairs[0]["chosen"], second)
        self.assertEqual(pairs[0]["rejected"], first)

    def test_other_conversation_makes_no_pair(self):
        r = cap.Regenerates()
        self.assertEqual(r.observe("a", "conv-1", ids(CONV_A + [40]), ids([50, 51, 52, 53])), [])
        self.assertEqual(r.observe("b", "conv-2", ids(CONV_B + [40]), ids([50, 51, 52, 53])), [])
        # conv-2's prompt contains conv-1's answer: no pair, because the conversation differs
        self.assertEqual(r.observe("c", "conv-2", ids(CONV_B + [50, 51, 52, 53]), ids([1])), [])
        self.assertEqual(r.observe("d", None, ids(CONV_A + [40]), ids([9])), [])

    def test_id_runs_must_be_whole_ids(self):
        # the bytes of the id run [1 2] occur at offset 2, which is not an id boundary: not a match
        hay = b"\x00\x00" + ids([1, 2]) + b"\x00\x00"
        self.assertFalse(cap._ids_contain(hay, ids([1, 2])))
        self.assertTrue(cap._ids_contain(ids([5, 1, 2, 9]), ids([1, 2])))
        self.assertFalse(cap._ids_contain(ids([5]), b""))


class Queue(TempFolder):
    def test_a_full_queue_drops_and_counts_and_never_blocks(self):
        c = cap.Capture(self.dir, "fp", queue_max=2, start=False)     # no writer: the queue fills
        t0 = time.time()
        results = [c.request(CONV_A, [1], "stop", {}) for _ in range(5)]
        self.assertLess(time.time() - t0, 2.0)
        self.assertEqual(results, [True, True, False, False, False])
        self.assertEqual(c.dropped, 3)
        self.assertEqual(c.status(), "capture: 0 KB written, 3 dropped")
        c.close()


class Server(TempFolder):
    def test_a_served_request_is_recorded_and_off_writes_nothing(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hello there", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}).encode()
        with contextlib.closing(urllib.request.urlopen(urllib.request.Request(
                base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=30)) as r:
            self.assertEqual(r.status, 200)
        self.assertIsNone(svc.capture)                       # off: nothing was set up, nothing written
        svc.capture = cap.Capture(self.dir, "fp", model="m", turn_id=TURN)
        with contextlib.closing(urllib.request.urlopen(urllib.request.Request(
                base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=30)) as r:
            self.assertEqual(r.status, 200)
        svc.capture.close()
        recs = list(capture_read.iter_records(self.dir, kinds={"A"}))
        self.assertEqual(len(recs), 1)
        self.assertGreater(recs[0]["meta"]["n_prompt"], 0)
        self.assertGreater(recs[0]["meta"]["n_answer"], 0)
        self.assertEqual(len(recs[0]["arrays"]["prompt"]), recs[0]["meta"]["n_prompt"])
        self.assertEqual(len(recs[0]["arrays"]["answer"]), recs[0]["meta"]["n_answer"])


if __name__ == "__main__":
    unittest.main()
