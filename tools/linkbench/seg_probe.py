"""seg_probe.py - does the stand-in's segment tracker see the real tokens? (SPEC 17, 10 Oct 2026)

Sends three requests with `strata_standin_floors`: a request that thinks, a request answered without thinking that asks for code,
and a maths request (no code), then prints each request's answer head so it can be matched with the engine log's
`SEG token ... -> thinking T, fence F` lines (the server must run with STRATA_TRACE_SEG=1). Expected: the thinking request moves
to `thinking 0` at token 248069 (</think>); the code request toggles `fence 1` / `fence 0` at 71093 (or 52451) around its block;
the maths request makes no SEG line.

usage: seg_probe.py URL [FLOORS=2,2,4]
"""
import json, sys, urllib.request

url = sys.argv[1]
floors = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,2,4").split(",")]


def ask(prompt, think):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1500, "temperature": 0, "stream": False,
            "strata_standin_floors": floors}
    if not think:
        body.update({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or "")[:120].replace("\n", " "), (m.get("content") or "")[:200].replace("\n", " ")


for name, prompt, think in [
        ("thinking request", "What is 17 times 23? Think it through, then answer.", True),
        ("code request, no thinking", "Write a Python function that reverses a string. Reply with the function in a code block.", False),
        ("maths request, no thinking", "A train goes 60 km in 1.5 hours. What is its speed in km/h?", False)]:
    reasoning, content = ask(prompt, think)
    print(f"== {name}\n   reasoning head: {reasoning!r}\n   answer head:    {content!r}")
