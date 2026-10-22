"""An image job's GPU turn with parallel text requests: a request admitted after the job took the engine's control lines must
not wait for them while the job waits for it to finish (both hung until a client disconnected)."""
import threading
import time
import types
import unittest
from pathlib import Path

from serve import images as images_api
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service

TOK = ByteTokenizer()
END = TOK.encode("<|im_end|>", parse_special=True)


class BatchStub:
    """generate_batched reduced: take the control lines to be admitted, give them back, then decode in a slot."""
    batch, max_context, info, last = 2, 4096, {}, {}

    def __init__(self, decode_s):
        self.ctl, self.decode_s = threading.Lock(), decode_s

    def alive(self):
        return True

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        while not self.ctl.acquire(timeout=0.2):          # _take_control
            if cancel.is_set():
                return
            yield None
        self.ctl.release()
        t_end = time.monotonic() + self.decode_s
        while time.monotonic() < t_end:
            if cancel.is_set():
                return
            time.sleep(0.05)
            yield ord("x")
        yield END[0]


class ImageTurnParallel(unittest.TestCase):
    def test_no_deadlock_with_a_request_admitted_after_the_job(self):
        svc = Service(BatchStub(decode_s=0.8), TOK, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        log, cancels = [], [threading.Event() for _ in range(3)]

        def text(name, cancel):
            for kind, x in svc.run(TOK.encode("hi"), False, None, 400, {}, cancel):
                if kind == "done":
                    log.append((name, x["finish"]))

        def image():
            gen = types.SimpleNamespace(cfg={"need_mib": 1}, policy="keep", reload="lazy", _say=print)
            try:
                with images_api.gpu_turn(svc, gen, {"width": 64, "height": 64}, cancels[2]):
                    log.append(("image", "ran"))
            except Exception as e:      # noqa: BLE001
                log.append(("image", type(e).__name__))

        threads = [threading.Thread(target=text, args=("A", cancels[0]), daemon=True)]
        threads[0].start()
        time.sleep(0.3)                                    # A decodes in its slot
        threads.append(threading.Thread(target=image, daemon=True))
        threads[1].start()
        time.sleep(0.3)                                    # the job waits for A
        threads.append(threading.Thread(target=text, args=("B", cancels[1]), daemon=True))
        threads[2].start()
        for t in threads:
            t.join(8)
        try:
            self.assertIn(("image", "ran"), log)
            self.assertIn(("A", "stop"), log)
            self.assertIn(("B", "stop"), log)
        finally:
            for c in cancels:
                c.set()


if __name__ == "__main__":
    unittest.main()
