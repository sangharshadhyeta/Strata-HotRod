#!/usr/bin/env python3
"""Reasoning-accuracy check for GGML_CUDA_MOE_CACHE_SUBSTITUTE (docs/plan.md,
pending-work item 5: "Substitution: +4.7% guarded, quality bounded at +0.47%
perplexity. Needs a reasoning-benchmark check before shipping; off by
default.")

Perplexity bounds the AVERAGE damage of feeding a substituted (wrong) expert's
weights into a row the router actually wanted. It does not say whether that
damage concentrates in a way that breaks multi-step reasoning specifically -
a domain where one early wrong token compounds instead of averaging out, which
is exactly the failure mode perplexity (a per-token, order-independent metric)
is least suited to catch. This is why plan.md calls for a SEPARATE reasoning
check rather than treating the perplexity number as sufficient.

Method: run the same fixed set of grade-school arithmetic word problems (each
with a single verifiable integer answer) against a live server twice - once
with substitution off (env unset on the server process), once on - and compare
exact-match accuracy. Not a substitute for a large external benchmark suite;
a small, fast, targeted regression check answering the specific question this
plan item asks: does substitution measurably hurt step-by-step reasoning.

Usage:
    python3 moe-substitute-reasoning-bench.py --url http://localhost:8091
"""
import argparse
import json
import re
import sys
import urllib.request

# 24 short grade-school arithmetic word problems, each with one unambiguous
# integer final answer. Deliberately simple (no unit conversion ambiguity, no
# multiple valid phrasings of the answer) so grading is exact-match on the
# last integer in the response, not another LLM call.
PROBLEMS = [
    ("Sam has 3 boxes of 8 pencils each. He gives away 5 pencils. How many pencils does he have left?", 19),
    ("A train travels 60 miles per hour for 3 hours, then 40 miles per hour for 2 hours. How many total miles did it travel?", 260),
    ("Maria bought 4 shirts at $12 each and a pair of shoes for $35. How much did she spend in total?", 83),
    ("A baker makes 96 cookies and packs them into bags of 8. He sells 7 bags. How many cookies are left unsold?", 40),
    ("There are 28 students in a class. 3/4 of them passed the test. How many students passed?", 21),
    ("A tank holds 500 liters. It is currently 3/5 full. How many more liters are needed to fill it?", 200),
    ("Jake reads 15 pages per day. How many pages will he have read after 12 days?", 180),
    ("A store had 240 apples. It sold 3/8 of them on Monday and 50 more on Tuesday. How many apples are left?", 100),
    ("Two friends split a $84 restaurant bill evenly, then each leaves a $6 tip. How much did each person pay in total?", 48),
    ("A rectangular garden is 14 meters long and 9 meters wide. What is its area in square meters?", 126),
    ("A factory produces 450 widgets in 9 hours, at a constant rate. How many widgets does it produce in 5 hours?", 250),
    ("Lily has $150. She spends 2/5 of it on a bike and then $20 on a helmet. How much money does she have left?", 70),
    ("A bus has 48 seats. On one trip it is 3/4 full, and 6 more people board. How many people are now on the bus?", 42),
    ("A recipe needs 250 grams of flour per batch. How many grams of flour are needed for 7 batches?", 1750),
    ("Tom saved $18 per week for 10 weeks, then spent $95 on a game. How much money does he have left?", 85),
    ("A parking lot has 6 rows with 24 spaces each. If 89 spaces are occupied, how many are empty?", 55),
    ("A school orders 15 boxes of 40 pencils each for 3 classrooms, split evenly. How many pencils does each classroom get?", 200),
    ("A car's fuel tank holds 48 liters. It uses 6 liters per 100 km. How many km can it travel on a full tank?", 800),
    ("A movie ticket costs $11. A group of 6 friends buys tickets and shares a $18 popcorn bucket evenly. How much does each friend pay in total?", 14),
    ("A warehouse has 3200 boxes. 1/4 are shipped out, then 300 new boxes arrive. How many boxes are in the warehouse now?", 2700),
    ("An office orders 9 reams of paper at $7 each, plus a one-time $15 delivery fee. What is the total cost?", 78),
    ("A pool is being filled at 25 liters per minute. How many minutes does it take to fill a 3000-liter pool?", 120),
    ("A farmer has 17 rows of 12 corn plants each. 3 rows are destroyed by a storm. How many corn plants remain?", 168),
    ("A library has 540 books. It donates 1/3 of them and receives 45 new ones. How many books does it have now?", 405),
]

PROMPT_TEMPLATE = (
    "Solve this step by step, then give the final answer as a single integer "
    "on its own line starting with 'ANSWER:'.\n\n{question}\n"
)


def extract_answer(text: str):
    m = re.findall(r"ANSWER:\s*-?\$?([\d,]+)", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m[-1].replace(",", ""))
        except ValueError:
            pass
    # Fallback: last integer anywhere in the response.
    nums = re.findall(r"-?\d[\d,]*", text)
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


def query(url: str, question: str, n_predict: int, timeout: float) -> str:
    body = json.dumps({
        "prompt": PROMPT_TEMPLATE.format(question=question),
        "n_predict": n_predict,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["content"]


def run(url: str, n_predict: int, timeout: float, label: str):
    correct = 0
    details = []
    for i, (question, expected) in enumerate(PROBLEMS):
        content = query(url, question, n_predict, timeout)
        got = extract_answer(content)
        ok = got == expected
        correct += ok
        details.append((i, expected, got, ok))
        print(f"[{label}] {i+1:2d}/{len(PROBLEMS)} expected={expected:<6} got={str(got):<6} "
              f"{'OK' if ok else 'WRONG'}", file=sys.stderr)
    acc = correct / len(PROBLEMS)
    print(f"[{label}] accuracy: {correct}/{len(PROBLEMS)} = {acc:.3f}", file=sys.stderr)
    return acc, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8091")
    ap.add_argument("--n-predict", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    acc, details = run(args.url, args.n_predict, args.timeout, args.label)
    print(json.dumps({"label": args.label, "accuracy": acc, "n": len(PROBLEMS),
                       "correct": sum(1 for *_, ok in details if ok)}))


if __name__ == "__main__":
    main()
