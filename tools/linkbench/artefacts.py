"""Artefact scan of every saved answer (analysis of saved text, not part of any engine decision): repetition loops, repeated characters,
leaked control tokens, a stray opening backtick, a truncation at the length limit. Wrong answers are NOT artefacts (the user, 10 Oct 2026:
"damage is not the wrong answer but the artefacts like repetition loops"). Flags are for READING: every flagged answer is then read."""
import glob, json, os, sys

T = "/root/.claude/jobs/743230e4/tmp"


def loop_run(words, n):
    """The longest number of times one n-word phrase repeats back to back (a loop)."""
    best = 1
    for i in range(0, max(0, len(words) - 2 * n)):
        run, j = 1, i + n
        while j + n <= len(words) and words[j:j + n] == words[i:i + n]:
            run += 1
            j += n
        best = max(best, run)
    return best


def char_run(text):
    """The longest run of one letter or digit (markdown rules of - = * _ and spaces are not artefacts)."""
    best = cur = 1
    for a, b in zip(text, text[1:]):
        cur = cur + 1 if a == b and a.isalnum() else 1
        best = max(best, cur)
    return best


def flags(text):
    w = text.split()
    f = []
    if len(w) >= 12:
        for n in (1, 2, 3, 4, 6, 8):
            r = loop_run(w, n)
            if (n <= 2 and r >= 8) or (n >= 3 and r >= 4):
                f.append(f"loop: a {n}-word phrase repeats {r} times in a row")
                break
    if char_run(text) >= 12:
        f.append(f"character run of {char_run(text)}")
    for c in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>"):
        if c in text:
            f.append(f"leaked control token {c}")
    if text.startswith("`") and not text.startswith("```"):
        f.append("stray opening backtick")
    return f


def scan(label, rows):
    flagged = [(r, flags(r[1])) for r in rows if flags(r[1])]
    return label, len(rows), flagged


results = []
for arm in ("BC", "B2", "E1", "E2", "P1", "P2", "R1", "P1b"):
    p = f"{T}/bench_{arm}.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    rows = [(f"gsm8k #{x['i']}", x["text"]) for x in d["gsm8k"]] + [(x["task"], x["text"]) for x in d["humaneval"]]
    results.append(scan(f"bench {arm}", rows))
for p in sorted(glob.glob(f"{T}/tps_*_texts.json")):
    arm = os.path.basename(p)[4:-11]
    d = json.load(open(p))
    rows = [(f"{k} #{i}", t) for k, v in d.items() for i, t in enumerate(v)]
    results.append(scan(f"speed prompts {arm}", rows))
for label, n, flagged in results:
    if flagged or "--all" in sys.argv:
        print(f"{label}: {n} answers, {len(flagged)} flagged")
        for (name, text), fl in flagged:
            print(f"    {name} ({len(text.split())} words): {'; '.join(fl)}")
print("\nclean (nothing flagged):", ", ".join(l for l, n, f in results if not f))
