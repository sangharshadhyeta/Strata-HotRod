"""Inkling 2's typed segments through InklingParser + OutputParser (serve/inkling.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402
from serve.inkling import InklingParser, inkling_kwargs, inkling_messages  # noqa: E402

THINK, TEXT, CALL, MODEL, END, EOS = ("<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|message_model|>",
                                      "<|end_message|>", "<|content_model_end_sampling|>")


def run(text: str, step: int, thinking: bool = True, tools=None):
    p = InklingParser(OutputParser(thinking=thinking, tools=tools, stream_tools=False), thinking)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return evs


def collect(evs):
    reasoning = "".join(e.text or "" for e in evs if e.kind == "reasoning")
    content = "".join(e.text or "" for e in evs if e.kind == "content")
    calls = [e for e in evs if e.kind == "tool_call"]
    return reasoning, content, calls


ANSWER = f"{MODEL}{THINK}The user asks the capital. It is Paris.{END}{MODEL}{TEXT}The capital of France is Paris.{END}{EOS}"


def test_reasoning_and_answer_split_at_every_chunk_size():
    for step in (1, 2, 3, 7, 1000):
        r, c, calls = collect(run(ANSWER, step))
        assert r == "The user asks the capital. It is Paris.", (step, r)
        assert c == "The capital of France is Paris.", (step, c)
        assert not calls


def test_plain_answer_without_segments():
    r, c, _ = collect(run("Hello there.", 4, thinking=False))
    assert r == "" and c == "Hello there."


def test_tool_call():
    text = f'{MODEL}{THINK}Need the weather.{END}{MODEL}get_weather{CALL}{{"name": "get_weather", "args": {{"city": "Paris", "days": 3}}}}{END}{EOS}'
    tools = [{"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}]
    for step in (1, 5, 1000):
        r, c, calls = collect(run(text, step, tools=tools))
        assert r == "Need the weather.", (step, r)
        assert len(calls) == 1, (step, calls)
        call = calls[0].call
        assert call.name == "get_weather" and call.arguments == {"city": "Paris", "days": 3}, (step, call)


def test_messages_and_kwargs():
    m = inkling_messages([{"role": "developer", "content": None}, {"role": "assistant", "content": "x", "tool_calls": [
        {"function": {"name": "f", "arguments": '{"a": 1}'}}]}])
    assert m[0]["role"] == "system" and m[0]["content"] == ""
    assert m[1]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert inkling_kwargs({"enable_thinking": False}) == {"reasoning_effort": "none"}
    assert inkling_kwargs({"reasoning_effort": "high"}) == {"reasoning_effort": "high"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
