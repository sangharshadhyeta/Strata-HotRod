"""bench.py - accuracy of a running Strata server on GSM8K and HumanEval (10 Oct 2026), for the stand-in / speed trade-off.

Why these two: a one-letter multiple-choice answer is produced by the prompt pass (exact routing), so it would never touch the
stand-ins; both of these make the model decode a long answer. Greedy (temperature 0). Same questions for every arm
(fetch_bench.py). Each question's answer, correctness and time are saved so the answers can be READ: a score is a screen,
never the verdict (SPEC.md 15).

usage: bench.py URL NAME REPEATS MAX_TOKENS     (called by strata_ab.py when AB_TPS points here; REPEATS and MAX_TOKENS ignored)
env:   BENCH_SETS (default /root/.claude/jobs/743230e4/tmp/bench_sets.json), BENCH_N_GSM, BENCH_N_HE (use the first N)
"""
import json, os, subprocess, sys, tempfile, time, urllib.request

url, name = sys.argv[1], sys.argv[2]
SETS = os.environ.get("BENCH_SETS", "/root/.claude/jobs/743230e4/tmp/bench_sets.json")
OUT = f"/root/.claude/jobs/743230e4/tmp/bench_{name}.json"   # the arm name decides the file: a thinking run needs its own arm name
sets = json.load(open(SETS))
n_gsm = int(os.environ.get("BENCH_N_GSM", len(sets["gsm8k"])))
n_he = int(os.environ.get("BENCH_N_HE", len(sets["humaneval"])))


THINK = os.environ.get("BENCH_THINK") == "1"   # the model thinks (inside <think>) before it answers: the reasoning task, scored on the final answer


NO_CAP = 30000   # no limit on an answer's length except the context window (32,768 tokens): the user's rule, an answer ends when the model ends it


def ask(prompt, max_tokens=None):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": NO_CAP, "temperature": 0, "stream": False}
    if not THINK:
        body.update({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    # the whole reply is kept (10 Oct 2026: "are these really empty?"): the server files reasoning text under `reasoning_content` and
    # tool calls under `tool_calls`, so an empty `content` is not yet an empty answer
    global LAST_RAW
    LAST_RAW = {"message": d["choices"][0]["message"], "finish_reason": d["choices"][0].get("finish_reason"), "usage": d.get("usage")}
    return d["choices"][0]["message"]["content"] or "", (d.get("usage") or {}).get("completion_tokens", 0), time.time() - t0


LAST_RAW = None


def number_after_answer(text):
    """The number on the last 'Answer:' line (digits, one sign, one point; thousands commas dropped)."""
    k = text.rfind("Answer:")
    if k < 0:
        return None
    s, out = text[k + len("Answer:"):].strip(), ""
    for ch in s:
        if ch.isdigit() or (ch in "-." and not out.count(ch)) or ch == ",":
            if ch != ",":
                out += ch
        elif out:
            break
    try:
        return float(out.rstrip("."))
    except ValueError:
        return None


def gsm_truth(answer):
    return float(answer.split("####")[-1].strip().replace(",", ""))


def code_block(text):
    k = text.find("```python")
    if k < 0:
        k = text.find("```")
        if k < 0:
            return text
        k += 3
    else:
        k += len("```python")
    e = text.find("```", k)
    return text[k:e if e >= 0 else len(text)]


def run_tests(prompt, code, test, entry):
    head = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
    prog = f"{head}\n{code}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        open(path, "w").write(prog)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=20)
            return p.returncode == 0, p.stderr[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


res = {"gsm8k": [], "humaneval": []}
tok = sec = 0.0
for i, q in enumerate(sets["gsm8k"][:n_gsm]):
    text, n, dt = ask(q["question"] + "\n\nSolve it step by step, then finish with a final line of the form: Answer: <number>", 700)
    got, want = number_after_answer(text), gsm_truth(q["answer"])
    ok = got is not None and abs(got - want) < 1e-6
    res["gsm8k"].append({"i": i, "ok": ok, "got": got, "want": want, "tokens": n, "s": round(dt, 1), "text": text, "raw": LAST_RAW})
    tok += n
    sec += dt
    print(f"gsm8k {i + 1}/{n_gsm} {'ok' if ok else 'WRONG'} (want {want}, got {got}) {n} tokens {dt:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"))
for i, q in enumerate(sets["humaneval"][:n_he]):
    text, n, dt = ask("Complete the following Python function. Reply with the complete function in one ```python code block.\n\n" + q["prompt"], 700)
    ok, err = run_tests(q["prompt"], code_block(text), q["test"], q["entry_point"])
    res["humaneval"].append({"i": i, "task": q["task_id"], "ok": ok, "err": err, "tokens": n, "s": round(dt, 1), "text": text, "raw": LAST_RAW})
    tok += n
    sec += dt
    print(f"humaneval {i + 1}/{n_he} {q['task_id']} {'pass' if ok else 'FAIL'} {n} tokens {dt:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"))
g = sum(x["ok"] for x in res["gsm8k"])
h = sum(x["ok"] for x in res["humaneval"])
print(f"BENCH {name}: gsm8k {g}/{len(res['gsm8k'])}  humaneval {h}/{len(res['humaneval'])}  "
      f"{tok / max(sec, 1e-9):.1f} answer tokens/s overall (prompt time included)", flush=True)
