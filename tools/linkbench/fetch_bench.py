"""fetch_bench.py - the fixed question sets for bench.py (10 Oct 2026): GSM8K (grade-school maths, ground-truth number) and
HumanEval (code, run against the benchmark's own tests), both from the Hugging Face datasets server. The sample is fixed by a
stride, so every arm sees the same questions.  Written once to OUT (default: bench_sets.json next to this script's job dir).

usage: fetch_bench.py OUT_JSON [N_GSM8K=100] [N_HUMANEVAL=40]
"""
import json, sys, time, urllib.error, urllib.request

BASE = "https://datasets-server.huggingface.co/rows"


def rows(dataset, config, split, offset, length):
    url = f"{BASE}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(5 * (attempt + 1))   # rate limited: back off
    raise RuntimeError("datasets server kept answering 429")


def strided(dataset, config, split, total, n):
    """n rows spread evenly over `total`: the whole split is read in pages of 100 (the server's cap), then sampled."""
    allrows = []
    for off in range(0, total, 100):
        allrows += rows(dataset, config, split, off, min(100, total - off))
        time.sleep(2)
    step = max(1, len(allrows) // n)
    return [allrows[i * step] for i in range(n)]


out_path = sys.argv[1]
n_gsm = int(sys.argv[2]) if len(sys.argv) > 2 else 100
n_he = int(sys.argv[3]) if len(sys.argv) > 3 else 40
gsm = strided("openai/gsm8k", "main", "test", 1319, n_gsm)
he = strided("openai/openai_humaneval", "openai_humaneval", "test", 164, n_he)
json.dump({"gsm8k": gsm, "humaneval": he}, open(out_path, "w"))
print(f"gsm8k {len(gsm)} questions, humaneval {len(he)} problems -> {out_path}")
