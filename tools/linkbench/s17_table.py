"""s17_table.py - the SPEC 17 HumanEval runs side by side (SPEC.md 18.1), rebuilt from the per-question result files.

usage: s17_table.py [SPEC.md]   Reads /root/.claude/jobs/743230e4/tmp/bench_<arm>.json (HumanEval part) for the arms below, and
replaces what stands between the S17-TABLE markers. For every answer without visible `content` it reports what the saved raw reply
held (only the arms run after d0651db keep it): reasoning text, tool calls, or nothing.
"""
import json, os, sys

T = "/root/.claude/jobs/743230e4/tmp"
SPEC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "SPEC.md")
START, END = "<!-- S17-TABLE-START -->", "<!-- S17-TABLE-END -->"
ARMS = [("BC", "control (nothing replaced)", "old binary"),
        ("B2", "ranks 2-9", "old binary"),
        ("E1", "2-9 + rank 1 within 1 logit", "old binary"),
        ("E2", "2-9 + rank 1 within 2 logits", "old binary"),
        ("R1", "ranks 2-9, flag off, no floors in the request (B2's setup)", "SPEC 17 binary"),
        ("P2", "floors 2,2,2, parts flag on", "SPEC 17 binary"),
        ("P1", "floors 2,2,4, parts flag on", "SPEC 17 binary"),
        ("P1b", "floors 2,2,4 again, whole reply kept", "SPEC 17 binary")]
rows = ["| Arm | Setting | Binary | HumanEval pass | Unfenced answers | No visible content | Whole-request answer tok/s | State |",
        "|---|---|---|---:|---:|---|---:|---|"]
notes = []
for arm, what, binary in ARMS:
    f = f"{T}/bench_{arm}.json"
    if not os.path.exists(f):
        rows.append(f"| {arm} | {what} | {binary} | | | | | not run |")
        continue
    h = json.load(open(f))["humaneval"]
    done = os.path.exists(f"{T}/ab_result_{arm}.json") or arm in ("BC", "B2", "E1", "E2")
    nocont = [x for x in h if not x["text"].strip()]
    rate = sum(x["tokens"] for x in h) / max(1e-9, sum(x["s"] for x in h))
    rows.append(f"| {arm} | {what} | {binary} | {sum(x['ok'] for x in h)} / {len(h)} | {sum('```' not in x['text'] for x in h)} | "
                f"{', '.join(x['task'].split('/')[1] for x in nocont) or 'none'} | {rate:.1f} | {'done' if done else 'running'} |")
    for x in nocont:
        raw = x.get("raw")
        if raw is None:
            notes.append(f"{arm} {x['task']}: {x['tokens']} tokens, no `content`; the raw reply was not kept (run before d0651db): UNVERIFIED")
        else:
            m = raw["message"]
            notes.append(f"{arm} {x['task']}: {x['tokens']} tokens, finish `{raw['finish_reason']}`; reasoning_content {len((m.get('reasoning_content') or '').split())} words, "
                         f"tool_calls {len(m.get('tool_calls') or [])}, content {len((m.get('content') or '').split())} words; "
                         f"reasoning head: {(m.get('reasoning_content') or '')[:160]!r}")
table = "\n".join(rows) + ("\n\n" + "\n".join("- " + n for n in notes) if notes else "")
text = open(SPEC).read()
a, b = text.index(START), text.index(END)
open(SPEC, "w").write(text[:a + len(START)] + "\n" + table + "\n" + text[b:])
print(table)
