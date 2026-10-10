"""bench_table.py - rebuilds the benchmark scores table in SPEC.md (section 18) from the sweep's result files and the engine log.

usage: bench_table.py [SPEC.md]   (default: SPEC.md at the repository root)
Reads /root/.claude/jobs/743230e4/tmp/bench_<arm>.json (written per question by bench.py), ab_result_bench.json (the first chain: arm BC)
and ab_result_bench2.json (the second chain), plus the engine log, whose per-request lines give the time to the first token (prompt
reading) and the decode speed; replaces what stands between the two markers. A finished arm is one that has an entry in a result file.
"""
import json, os, re, statistics, sys

T = "/root/.claude/jobs/743230e4/tmp"
LOG = "/home/Projects/Strata/strata-iq3_s.log"
MARK = "strata verify: window up to"
SPEC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "SPEC.md")
START, END = "<!-- BENCH-TABLE-START -->", "<!-- BENCH-TABLE-END -->"
# arm, what it replaces (ranks lo-9 are replaced, with the router's top picks 0..lo-1 exact; "edge" = a fractional floor)
ARMS = [("BC", "control (nothing replaced)"), ("B2", "ranks 2-9"),
        ("E1", "2-9, rank 1 only by a substitute within 1 logit (the edge, margin 1)"),
        ("E2", "2-9, rank 1 only within 2 logits (edge margin 2)"), ("E4", "2-9, rank 1 only within 4 logits (edge margin 4)"),
        ("B1", "ranks 1-9 (rank 1 at any distance)"), ("BC2", "control, second copy (the noise floor)"), ("B0", "ranks 0-9"),
        ("B6", "ranks 6-9"), ("B4", "ranks 4-9"), ("B9", "rank 9 only"), ("B8", "ranks 8-9"), ("B7", "ranks 7-9"),
        ("B5", "ranks 5-9"), ("B3", "ranks 3-9")]
done, chain2 = {}, []
for p in (f"{T}/ab_result_bench.json", f"{T}/ab_result_bench2.json"):
    if os.path.exists(p):
        for d in json.load(open(p)):
            done[d["arm"]] = d
            if p.endswith("bench2.json"):
                chain2.append(d["arm"])

req_re = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in (\d+) ms.*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")


def log_split():
    """arm -> (median time to first token s, median decode tok/s) for requests of 100+ tokens, from the engine log. The log holds one
    section per server start, in order: ..., BC, then the second chain's arms (the last one may be running)."""
    if not os.path.exists(LOG):
        return {}
    sections = open(LOG, errors="replace").read().split(MARK)[1:]
    started = re.findall(r"^=== arm (\w+)", open(f"{T}/ab_run_bench2.log").read(), re.M) if os.path.exists(f"{T}/ab_run_bench2.log") else []
    running = started[-1] if started and started[-1] not in chain2 else None
    end = len(sections) - (1 if running else 0)
    by_arm = {}
    for i, a in enumerate(chain2):
        by_arm[a] = sections[end - len(chain2) + i]
    # the old chain's killed B2 never reached the engine's marker, so BC sits directly before the second chain's first arm
    if "BC" in done and end - len(chain2) - 1 >= 0:
        by_arm["BC"] = sections[end - len(chain2) - 1]
    if running:
        by_arm[running] = sections[-1]
    out = {}
    for a, sec in by_arm.items():
        rows = [(int(m.group(2)) / 1000.0, float(m.group(5))) for l in sec.splitlines()
                for m in [req_re.search(l)] if m and int(m.group(3)) >= 100]
        if rows:
            out[a] = (statistics.median(r[0] for r in rows), statistics.median(r[1] for r in rows))
    return out


split = log_split()
rows = ["| Arm | Replaced | GSM8K correct | HumanEval pass | Decode tok/s (median) | First token (s, median) | Whole-request tok/s | Longest answer (tokens) | State |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
for arm, what in ARMS:
    f = f"{T}/bench_{arm}.json"
    if not os.path.exists(f):
        rows.append(f"| {arm} | {what} | | | | | | | pending |")
        continue
    d = json.load(open(f))
    g, h = d["gsm8k"], d["humaneval"]
    tok = sum(x["tokens"] for x in g + h)
    sec = sum(x["s"] for x in g + h)
    finished = arm in done and not done[arm].get("error")
    state = "done" if finished else ("void: " + str(done[arm].get("error") or done[arm].get("memory_guard")) if arm in done else "running")
    longest = max([x["tokens"] for x in g + h] or [0])
    ttft, dec = split.get(arm, (None, None))
    rows.append(f"| {arm} | {what} | {sum(x['ok'] for x in g)} / {len(g)} | {sum(x['ok'] for x in h)} / {len(h)} | "
                f"{'' if dec is None else f'{dec:.1f}'} | {'' if ttft is None else f'{ttft:.1f}'} | "
                f"{tok / max(sec, 1e-9):.1f} | {longest} | {state} |")
table = "\n".join(rows)
text = open(SPEC).read()
a, b = text.index(START), text.index(END)
open(SPEC, "w").write(text[:a + len(START)] + "\n" + table + "\n" + text[b:])
print(table)
