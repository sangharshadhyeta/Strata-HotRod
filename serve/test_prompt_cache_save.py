"""The prompt cache's save runs after the reply's end is out, and still holds the turn until it is written."""
import threading
import time
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service


class SlowCache:
    """Stands in for PromptCache: before() restores nothing, after() takes as long as a big save."""

    def __init__(self, svc, save_s):
        self.svc, self.save_s = svc, save_s
        self.held = None
        self.saved_at = None
        self.saves = []                           # the end time of each save, in order
        self.fifo_held_during_save = None

    def before(self, engine, ids):
        return None

    def after(self, engine, ids, generated, prompt_read, save=True):
        self.fifo_held_during_save = self.svc.fifo.locked()
        time.sleep(self.save_s)
        self.saved_at = time.monotonic()
        self.saves.append(self.saved_at)
        return None


class PassEngine(MockEngine):
    """MockEngine plus the DONE record of each pass (`last`), which the prompt cache's save waits for."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:                                  # the server closes the pass at its stop token
            self.last = {"generated": n, "prompt_read": len(ids)}


class PromptCacheSaveOrder(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.tok = tok
        self.svc = Service(PassEngine(tok, "Hello."), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.cache = self.svc.pcache = SlowCache(self.svc, 0.5)

    def test_done_before_save_and_turn_held(self):
        t0 = time.monotonic()
        kinds = []
        for kind, _ in self.svc.run(self.tok.encode("hi"), False, None, 8, {}, threading.Event()):
            kinds.append(kind)
        ended = time.monotonic()
        self.assertEqual(kinds[-1], "done")
        self.assertLess(ended - t0, 0.4, "the reply waited for the save")
        self.assertTrue(self.svc.fifo.locked(), "the turn was released before the save finished")
        self.assertTrue(self.svc.fifo.acquire(timeout=5), "the save never released the turn")
        self.svc.fifo.release()
        self.assertIsNotNone(self.cache.saved_at)
        self.assertTrue(self.cache.fifo_held_during_save)

    def test_next_request_waits_for_the_save(self):
        list(self.svc.run(self.tok.encode("hi"), False, None, 8, {}, threading.Event()))
        engine, started = self.svc.engine, []
        orig = engine.generate

        def timed(*a, **k):
            started.append(time.monotonic())
            return orig(*a, **k)
        engine.generate = timed
        list(self.svc.run(self.tok.encode("hi"), False, None, 8, {}, threading.Event()))
        self.assertTrue(self.cache.saves)
        first_saved = self.cache.saves[0]
        self.assertTrue(started)
        self.assertGreaterEqual(started[0], first_saved, "the next request ran on the engine during the save")


if __name__ == "__main__":
    unittest.main()
