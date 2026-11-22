"""serve/inkling.py - Inkling 2 (Thinking Machines) on HotRod's server.

Inkling writes typed segments instead of tags: `<|message_model|><|content_thinking|>...<|end_message|>` is reasoning, `<|message_model|><|content_text|>...` is the
answer, and a tool call is `<|message_model|>NAME<|content_invoke_tool_json|>{"name": .., "args": {..}}<|end_message|>`; the turn ends at
`<|content_model_end_sampling|>`.  `InklingParser` rewrites that stream, as it arrives, into what `OutputParser` already reads (`</think>` to close the reasoning, the
`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` form), then hands it to the ordinary parser, so streaming, tool-call events, stop strings
and usage accounting all stay the server's own.

The segment grammar and the idea of rewriting it ahead of the parser come from Project Maya's serve/frontend.py (`InklingRewriter`, MIT, see
third_party/project-maya/NOTICE.md); this version is written for our OutputParser (no THINK_AGAIN: a reasoning segment after the answer has begun is shown as answer text).
"""
from __future__ import annotations

import copy
import json

INK_THINK, INK_TEXT, INK_CALL = "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>"
INK_MODEL, INK_END, INK_EOS = "<|message_model|>", "<|end_message|>", "<|content_model_end_sampling|>"
INK_OTHER_TEXT = ("<|content_tool_error|>", "<|content_invoke_tool_text|>")
INKLING_EFFORT = ("none", "minimal", "low", "medium", "high", "xhigh", "max")   # the template's effort_map


def is_inkling_template(source: str) -> bool:
    return "<|content_invoke_tool_json|>" in source and "effort_map" in source


def inkling_messages(messages: list[dict]) -> list[dict]:
    """The messages as Inkling's template takes them: a tool call's arguments an object (it refuses a JSON string), no None content,
    a developer message a system one, reasoning as reasoning_content."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("reasoning") and not m.get("reasoning_content"):
            m["reasoning_content"] = m["reasoning"]
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = copy.deepcopy(tc)
                fn = tc.setdefault("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except ValueError:
                        fn["arguments"] = {"_raw": args}
                elif args is None:
                    fn["arguments"] = {}
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


def inkling_kwargs(kwargs: dict) -> dict:
    """The server's thinking kwargs in the template's words: enable_thinking=False is effort "none"."""
    kw = dict(kwargs)
    if kw.get("enable_thinking") is False:
        kw.pop("enable_thinking")
        kw["reasoning_effort"] = "none"
    return kw


class InklingRewriter:
    """Inkling's output, as it streams, into what OutputParser reads.  A mark split across deltas is held until it is whole; what stands between segments (the
    tool's name after <|message_model|>, <|end_message|>) is dropped."""

    MARKS = (INK_THINK, INK_TEXT, INK_CALL, INK_MODEL, INK_END, INK_EOS) + INK_OTHER_TEXT

    def __init__(self, thinking: bool = True):
        self.buf = ""
        self.mode = "lead"           # lead (before any segment) | think | text | name | call | between
        self.open = thinking         # the parser is in its reasoning state (it starts there when thinking)
        self.name = ""
        self.body = ""

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while self.buf:
            hits = [(i, m) for m in self.MARKS for i in (self.buf.find(m),) if i >= 0]
            if not hits:
                keep = max((n for m in self.MARKS for n in range(1, len(m)) if self.buf.endswith(m[:n])), default=0)
                out.append(self._text(self.buf[:len(self.buf) - keep]))
                self.buf = self.buf[len(self.buf) - keep:]
                break
            i, m = min(hits)
            if i:
                out.append(self._text(self.buf[:i]))
            self.buf = self.buf[i + len(m):]
            out.append(self._mark(m))
        return "".join(out)

    def _text(self, s: str) -> str:
        if self.mode == "name":
            self.name += s
            return ""
        if self.mode == "call":
            self.body += s
            return ""
        if self.mode == "lead":      # no segment opened: a model that answers plainly
            if not s.strip():
                return ""
            self.mode = "text"
            return self._close() + s
        return s if self.mode in ("think", "text") else ""

    def _close(self) -> str:
        if self.open:
            self.open = False
            return "</think>"
        return ""

    def _mark(self, m: str) -> str:
        if m == INK_THINK:
            self.mode = "think"
            return ""                # (our parser starts in reasoning when thinking; a later reasoning segment reads as answer text)
        if m == INK_TEXT or m in INK_OTHER_TEXT:
            self.mode = "text"
            return self._close()
        if m == INK_MODEL:
            self.mode, self.name = "name", ""
            return ""
        if m == INK_CALL:
            self.mode, self.body = "call", ""
            return ""
        out = self._call() if self.mode == "call" else ""
        self.mode = "between"        # <|end_message|> / the end of sampling: the segment is over
        return out

    def finish(self) -> str:
        """The end of the reply: a mark's partial text held back is text; a call still open goes out as far as it is whole."""
        rest, self.buf = self.buf, ""
        out = self._text(rest)
        if self.mode == "call":
            out += self._call()
            self.mode = "between"
        return out

    def _call(self) -> str:
        try:
            j = json.loads(self.body)
            name = (j.get("name") if isinstance(j, dict) else None) or self.name.strip()
            args = (j.get("args") if isinstance(j, dict) else None) or {}
            if not name or not isinstance(args, dict):
                raise ValueError(self.body)
        except ValueError:
            return self._close() + self.body     # not a call the parser can read: its text, as the answer's
        val = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        body = "".join(f"<parameter={k}>\n{val(v)}\n</parameter>\n" for k, v in args.items())
        return self._close() + f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class InklingParser:
    """OutputParser behind the rewriter: the same feed / finish / state / buf the server uses."""

    def __init__(self, parser, thinking: bool = True):
        self._p = parser
        self._r = InklingRewriter(thinking)

    def feed(self, delta: str):
        return self._p.feed(self._r.feed(delta))

    def finish(self, reason=None):
        tail = self._r.finish()
        evs = self._p.feed(tail) if tail else []
        return evs + self._p.finish(reason)

    def __getattr__(self, name):         # state, buf, ... : the parser's own
        return getattr(self._p, name)
