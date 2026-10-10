"""prefill_probe.py - time to the first token against prompt length (SPEC 19, prefill), for strata_ab.py (AB_TPS=this file).

usage: prefill_probe.py URL NAME REPEATS MAX_TOKENS     (MAX_TOKENS ignored)
Each request is a DIFFERENT slice of the project's own text (so no prompt shares a prefix with an earlier one: the engine reuses
a prompt prefix inside a session) asking for ONE token, so the request time is the prompt's read time plus one decode step.
Lengths come from PF_LENGTHS (default 100,400,1600 prompt tokens, as the server counts them). Prints prompt tokens, seconds and
prompt tok/s per request; the arm's result is also saved to prefill_<NAME>.json beside the other results.
"""
import json, os, sys, time, urllib.request

url, name, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3])
T = "/root/.claude/jobs/743230e4/tmp"
lengths = [int(x) for x in os.environ.get("PF_LENGTHS", "100,400,1600").split(",")]
text = open("/home/Projects/Strata/SPEC.md", encoding="utf-8").read()


def ask(prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0, "stream": False,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d["usage"]["prompt_tokens"], time.time() - t0


slice_no = [0]


def fresh(n_tokens):
    """A slice of about n_tokens tokens (about 4 characters each) from a place no earlier request used."""
    chars = n_tokens * 4
    start = (slice_no[0] * 7919 * 13) % max(1, len(text) - chars)
    slice_no[0] += 1
    return "Summarise this in one word:\n\n" + text[start:start + chars]


ask(fresh(50))   # warm-up: not timed
rows = {}
for n in lengths:
    rows[n] = []
    for _ in range(repeats):
        pt, s = ask(fresh(n))
        rows[n].append({"prompt_tokens": pt, "s": round(s, 2), "tok_per_s": round(pt / s, 1)})
        print(f"prompt {pt} tokens: {s:.2f} s, {pt / s:.1f} prompt tok/s", flush=True)
json.dump(rows, open(f"{T}/prefill_{name}.json", "w"), indent=1)
