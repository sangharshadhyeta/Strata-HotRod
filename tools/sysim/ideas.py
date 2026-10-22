"""The five borrowed ideas in sysim, each against today's setup and on top of the read-ahead fixes (mean of 3 seeds).
usage: python ideas.py [--windows 40] [--profile rtx3060] [--set key=value ...]   (the --set applies to both bases)"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysim  # noqa: E402

FIXED = dict(io_threads=1, yield_=1, edf=1, recall1=0.69, wrong_per_right=0.75 * (1 - 0.69) / (1 - 0.553), zero_copy=1)

IDEAS = [
    ("1 cost rule: confidence >= 0.5", dict(prefetch_theta=0.5)),
    ("1 cost rule: confidence >= 0.6", dict(prefetch_theta=0.6)),
    ("1 cost rule >= 0.5 + value order", dict(prefetch_theta=0.5, value_order=1)),
    ("1 ... if confidence separates poorly (sep 0.1)", dict(prefetch_theta=0.5, value_order=1, conf_sep=0.1)),
    ("1 ... if confidence separates well (sep 0.4)", dict(prefetch_theta=0.5, value_order=1, conf_sep=0.4)),
    ("2 multi-history: 3-ahead guesses, recall 0.3", dict(recall3=0.3)),
    ("2 multi-history: 3-ahead 0.5, 2-ahead 0.6", dict(recall3=0.5, recall2=0.6)),
    ("3 better caching: cold misses -25%", dict(cold_share=0.15)),
    ("3 better caching: cold misses -50%", dict(cold_share=0.10)),
    ("4 pacing: one guess start per 0.2 ms", dict(pace_s=0.2e-3)),
    ("4 pacing: one guess start per 0.5 ms", dict(pace_s=0.5e-3)),
    ("5 row-chunk streaming, 2 chunks", dict(chunks=2)),
    ("5 row-chunk streaming, 4 chunks", dict(chunks=4)),
    ("5 row-chunk streaming, 8 chunks", dict(chunks=8)),
    ("ALL: 1 (0.5, value) + 4 (0.2 ms) + 5 (4 chunks)", dict(prefetch_theta=0.5, value_order=1, pace_s=0.2e-3, chunks=4)),
    ("ALL + 2 (3-ahead 0.3) + 3 (-25%)", dict(prefetch_theta=0.5, value_order=1, pace_s=0.2e-3, chunks=4, recall3=0.3, cold_share=0.15)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    base = dict(sysim.PROFILES[a.profile])
    for kv in a.set:
        k, v = kv.split("=", 1)
        base[k] = type(base[k])(float(v)) if not isinstance(base[k], int) else int(float(v))
    fixed = dict(base, **FIXED)
    seeds = [1, 2, 3]
    t0, t1 = sysim.tok(base, a.windows, seeds), sysim.tok(fixed, a.windows, seeds)
    print(f"tokens/s (model, mean of 3 seeds): today {t0:.1f}; with the read-ahead fixes {t1:.1f} ({(t1 / t0 - 1) * 100:+.1f}%)\n")
    print(f"{'idea':52s} {'on today':>9} {'on the fixes':>13}")
    for label, ch in IDEAS:
        a0 = sysim.tok(dict(base, **ch), a.windows, seeds)
        a1 = sysim.tok(dict(fixed, **ch), a.windows, seeds)
        print(f"{label:52s} {(a0 / t0 - 1) * 100:+8.1f}% {(a1 / t1 - 1) * 100:+12.1f}%")


if __name__ == "__main__":
    main()
