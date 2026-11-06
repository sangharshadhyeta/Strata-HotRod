"""Warm reads (POST /v1/prefill and friends) on the DeepSeek server, with a fake engine (no GPU, no model pack).

    python -m pytest serve/test_warm.py -q
"""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, probe_messages, serve

TEMPLATE = Path(__file__).with_name("chat_template.jinja")
DRAFT = ("Here is a long draft question about the sea, the sky and the long road between them, written slowly "
         "while the user is still typing and not yet ready to send it to the model at all. ") * 3


class WarmEngine(MockEngine):
    """Records each generate() call; with a gate, it holds the turn until the gate is set (or the read is cancelled)."""

    def __init__(self, tok, gate=None):
        super().__init__(tok, "Answer.", max_context=32768)
        self.calls = []
        self.gate = gate

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        if self.gate is not None:
            while not self.gate.wait(0.01):
                if cancel.is_set():
                    return
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def make_service(engine, tok):
    return Service(engine, tok, ChatTemplate(TEMPLATE))


class WarmBase(unittest.TestCase):
    gate = None

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = WarmEngine(self.tok, self.gate)
        self.svc = make_service(self.engine, self.tok)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if self.gate is not None:
            self.gate.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=20)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode() or "{}")


class WarmEndpoints(WarmBase):
    def test_openai_prefill_answers_and_reads_one_token(self):
        code, j = self.post("/v1/prefill", {"model": "m", "messages": [{"role": "user", "content": "Hello there"}]})
        self.assertEqual(code, 200)
        self.assertEqual(j["object"], "prefill")
        self.assertNotIn("skipped", j)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.engine.calls[0][1], 1)              # max_new 1: a warm makes no answer

    def test_anthropic_prefill_routes_and_answers(self):
        code, j = self.post("/v1/messages/prefill", {"model": "m", "max_tokens": 50,
                                                     "messages": [{"role": "user", "content": "Hello there"}]})
        self.assertEqual((code, j["object"]), (200, "prefill"))
        code, j = self.post("/v1/prefill", {"model": "m", "system": "Be brief.",
                                            "messages": [{"role": "user", "content": "Hello there"}]})
        self.assertEqual((code, j["object"]), (200, "prefill"))
        self.assertEqual(len(self.engine.calls), 2)

    def test_chat_endpoint_with_flag_or_header_is_a_warm(self):
        body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "Hello there"}]}
        code, j = self.post("/v1/chat/completions", dict(body, strata_prefill=True))
        self.assertEqual((code, j["object"]), (200, "prefill"))
        code, j = self.post("/v1/messages", {"model": "m", "max_tokens": 9, "stream": False,
                                             "messages": [{"role": "user", "content": "Hi"}]},
                            headers={"X-Strata-Prefill": "1"})
        self.assertEqual((code, j["object"]), (200, "prefill"))
        self.assertEqual(len(self.engine.calls), 2)

    def test_client_cannot_set_the_warm_marker(self):
        code, j = self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "Hi"}],
                                                     "_strata_warm": {"ids": [], "cancel": 1}})
        self.assertEqual(code, 200)
        self.assertEqual(j["object"], "chat.completion")
        self.assertNotEqual(self.engine.calls[0][1], 1)           # not read as a warm (max_new 1)
        self.assertNotIn("_strata_warm", self.engine.calls[0][2])

    def test_prefill_warm_false_turns_it_off(self):
        self.svc.prefill_warm = False
        code, j = self.post("/v1/prefill", {"model": "m", "messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual((code, j["skipped"], j["reason"]), (200, True, "disabled"))
        self.assertEqual(self.engine.calls, [])

    def test_batched_engine_and_pictures_take_no_warm(self):
        self.engine.batch = 2
        code, j = self.post("/v1/prefill", {"model": "m", "messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual((j["skipped"], j["reason"]), (True, "parallel"))
        self.engine.batch = 0
        code, j = self.post("/v1/prefill", {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]})
        self.assertEqual((j["skipped"], j["reason"]), (True, "pictures"))
        self.assertEqual(self.engine.calls, [])


class WarmAccounting(WarmBase):
    def test_a_warm_is_not_in_history_totals_or_timings(self):
        code, j = self.post("/v1/prefill", {"model": "m", "messages": [{"role": "user", "content": DRAFT}]})
        self.assertEqual(j["object"], "prefill")
        self.assertEqual(len(self.svc.history), 0)
        self.assertEqual(self.svc.totals["requests"], 0)
        self.assertIsNone(self.svc.last_timings)
        code, j = self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(self.svc.totals["requests"], 1)           # a real request is counted as before
        self.assertEqual(len(self.svc.history), 1)


class WarmPin(WarmBase):
    def test_pin_is_where_the_draft_stable_text_ends(self):
        messages = [{"role": "user", "content": DRAFT}]
        code, j = self.post("/v1/prefill", {"model": "m", "messages": messages})
        self.assertEqual(j["object"], "prefill")
        ids = self.engine.calls[0][0]
        probe_ids, _, _ = self.svc.prepare(probe_messages(messages), None, {}, 1)
        common = 0
        for x, y in zip(ids, probe_ids):
            if x != y:
                break
            common += 1
        sampling = self.engine.calls[0][2]
        self.assertEqual(sampling["strata_prefix"], {"tokens": common - 1})
        self.assertLess(common - 1, len(ids) - 1)
        self.assertIn(f" pin={common - 1}", StrataEngine.sampling_keys(sampling))

    def test_no_warm_pin_when_the_request_has_a_shared_prefix(self):
        # DeepSeek-V4.1: prepare() sets the system prompt's shared start as the request's pin; the warm keeps it
        self.svc.family = "deepseek41"
        pinned = []
        original = self.svc.warm_pin_into

        def spy(*args, **kwargs):
            pinned.append(True)
            return original(*args, **kwargs)
        self.svc.warm_pin_into = spy
        code, j = self.post("/v1/messages/prefill", {"model": "m", "max_tokens": 9, "system": "Be brief " * 40,
                                                     "messages": [{"role": "user", "content": DRAFT}]})
        self.assertEqual(j["object"], "prefill")
        self.assertEqual(pinned, [])
        self.assertIn("strata_prefix", self.engine.calls[0][2])


class WarmScheduling(WarmBase):
    def setUp(self):
        self.gate = threading.Event()                  # the engine holds the turn until the test lets it go
        super().setUp()

    def wait_for(self, predicate, what, timeout=10.0):
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > end:
                self.fail(f"timed out waiting for {what}")
            time.sleep(0.01)

    def running_warm(self):
        return [w for w in self.svc.warms if w["running"]]

    def test_newer_warm_cancels_a_queued_one(self):
        # a running warm is kept while the next warms continue its ids (the draft typed on); a queued one a newer
        # warm supersedes is cancelled before it reads anything
        results = {}

        def go(name, ids):
            results[name] = self.svc.warm(ids, False, None, {"model": "m"}, threading.Event())
        a = threading.Thread(target=go, args=("a", [1, 2, 3]))
        a.start()
        self.wait_for(lambda: self.running_warm(), "the first warm to take the turn")
        b = threading.Thread(target=go, args=("b", [1, 2, 3, 4]))
        b.start()
        self.wait_for(lambda: len(self.svc.warms) == 2, "the second warm to queue")
        c = threading.Thread(target=go, args=("c", [1, 2, 3, 5]))
        c.start()
        self.wait_for(lambda: len(self.svc.warms) == 3 and self.svc.warms[1]["cancel"].is_set(),
                      "the newer warm to cancel the queued one")
        self.gate.set()
        for t in (a, b, c):
            t.join(15)
            self.assertFalse(t.is_alive())
        self.assertTrue(results["b"].get("skipped"), results["b"])
        self.assertNotIn("skipped", results["a"])
        self.assertNotIn("skipped", results["c"])

    def test_real_request_keeps_a_running_warm_it_continues_and_cancels_the_rest(self):
        running_a = {"ids": [1, 2, 3], "cancel": threading.Event(), "running": True}
        running_b = {"ids": [1, 2, 3, 4], "cancel": threading.Event(), "running": True}
        self.svc.warms.extend([running_a, running_b])
        self.svc._supersede([1, 2, 3, 4, 5])        # continues both running warms: both go on
        self.assertFalse(running_a["cancel"].is_set())
        self.assertFalse(running_b["cancel"].is_set())
        self.svc._supersede([9, 9, 9])              # continues neither: both are cancelled
        self.assertTrue(running_a["cancel"].is_set())
        self.assertTrue(running_b["cancel"].is_set())
        queued = {"ids": [1, 2, 3], "cancel": threading.Event(), "running": False}
        self.svc.warms.append(queued)
        self.svc._supersede([1, 2, 3, 4])           # a queued warm is cancelled even when the ids would continue it
        self.assertTrue(queued["cancel"].is_set())

    def test_real_request_through_run_cancels_a_queued_warm(self):
        # the same rule on the real path: a warm that holds nothing yet is cancelled by a request arriving behind it
        queued = {"ids": [7, 7, 7], "cancel": threading.Event(), "running": False}
        self.svc.warms.append(queued)
        self.gate.set()                              # the engine answers at once
        ids = [5, 6]
        list(self.svc.run(ids, False, None, 1, {"model": "m"}, threading.Event()))
        self.assertTrue(queued["cancel"].is_set())


if __name__ == "__main__":
    unittest.main()
