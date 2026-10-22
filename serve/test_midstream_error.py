"""An error after a stream began (GpuBusy, EngineStuck) ends the stream with its API's error event - never a second HTTP
response written into the event stream."""
import http.client
import json
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineStuck, GpuBusy, MockEngine, Service, serve

TOK = ByteTokenizer()


class Failing(MockEngine):
    exc = GpuBusy

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        yield from ()
        raise self.exc("the engine's batch slots are not free right now")


CASES = (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True}, "data: [DONE]"),
         ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, "stream": True}, "event: error"),
         ("/v1/responses", {"input": "hi", "stream": True}, "event: error"))


class MidstreamError(unittest.TestCase):
    def test_each_api_ends_with_its_error_event(self):
        for exc in (GpuBusy, EngineStuck):
            eng = Failing(TOK, "ok", max_context=4096)
            eng.exc = exc
            svc = Service(eng, TOK, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
            httpd = serve(svc, "127.0.0.1", 0)
            try:
                for path, body, end in CASES:
                    with self.subTest(exc=exc.__name__, path=path):
                        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
                        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                        r = c.getresponse()
                        text = r.read().decode()
                        self.assertEqual(r.status, 200)
                        self.assertNotIn("HTTP/1.", text)               # no status line inside the body
                        self.assertIn(end, text)
                        self.assertIn("batch slots are not free", text)
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
