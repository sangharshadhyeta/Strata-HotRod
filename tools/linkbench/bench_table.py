"""bench_table.py - rebuilds the benchmark scores table in SPEC.md (section 18) from the sweep's result files.

usage: bench_table.py [SPEC.md]   (default: SPEC.md at the repository root)
Reads /root/.claude/jobs/743230e4/tmp/bench_<arm>.json (written per question by bench.py) and ab_result_bench.json (written per arm
by strata_ab.py); replaces what stands between the two markers. A finished arm is one that has an entry in ab_result_bench.json.
"""
import json, os, sys

T = "/root/.claude/jobs/743230e4/tmp"
SPEC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "SPEC.md")
START, END = "<!-- BENCH-TABLE-START -->", "<!-- BENCH-TABLE-END -->"
# arm, what it replaces (ranks lo-9 are replaced, with the router's top picks 0..lo-1 exact; "edge" = a fractional floor)
ARMS = [("BC", "control (nothing replaced)"), ("B2", "ranks 2-9"),
        ("E1", "2-9, rank 1 only by a substitute within 1 logit (the edge, margin 1)"),
        ("E2", "2-9, rank 1 only within 2 logits (edge margin 2)"), ("E4", "2-9, rank 1 only within 4 logits (edge margin 4)"),
        ("B1", "ranks 1-9 (rank 1 at any distance)"), ("BC2", "control, second copy (the noise floor)"), ("B0", "ranks 0-9"),
        ("B6", "ranks 6-9"), ("B4", "ranks 4-9"), ("B9", "rank 9 only"), ("B8", "ranks 8-9"), ("B7", "ranks 7-9"),
        ("B5", "ranks 5-9"), ("B3", "ranks 3-9")]
done = {}
for p in (f"{T}/ab_result_bench.json", f"{T}/ab_result_bench2.json"):
    if os.path.exists(p):
        for d in json.load(open(p)):
            done[d["arm"]] = d
rows = ["| Arm | Replaced | GSM8K correct | HumanEval pass | Answer tok/s (prompt time included) | Longest answer (tokens) | State |",
        "|---|---|---:|---:|---:|---:|---|"]
for arm, what in ARMS:
    f = f"{T}/bench_{arm}.json"
    if not os.path.exists(f):
        rows.append(f"| {arm} | {what} | | | | | pending |")
        continue
    d = json.load(open(f))
    g, h = d["gsm8k"], d["humaneval"]
    tok = sum(x["tokens"] for x in g + h)
    sec = sum(x["s"] for x in g + h)
    finished = arm in done and not done[arm].get("error")
    state = "done" if finished else ("void: " + str(done[arm].get("error") or done[arm].get("memory_guard")) if arm in done else "running")
    longest = max([x["tokens"] for x in g + h] or [0])
    rows.append(f"| {arm} | {what} | {sum(x['ok'] for x in g)} / {len(g)} | {sum(x['ok'] for x in h)} / {len(h)} | "
                f"{tok / max(sec, 1e-9):.1f} | {longest} | {state} |")
table = "\n".join(rows)
text = open(SPEC).read()
a, b = text.index(START), text.index(END)
open(SPEC, "w").write(text[:a + len(START)] + "\n" + table + "\n" + text[b:])
print(table)
