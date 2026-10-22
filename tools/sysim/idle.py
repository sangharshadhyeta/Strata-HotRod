"""Idle time as waste: for each resource of the sysim timeline, how long it sat idle in a decode window and what the window was
waiting for at that moment (the critical-path task running then).  An idle stretch while the critical path runs on ANOTHER resource
is a bubble that only independent work can fill; idle while nothing is on the critical path does not exist (the path covers the window).
usage: python idle.py [--windows 40] [--profile rtx3060] [--set key=value ...]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysim  # noqa: E402


def intervals(tasks, lo, hi):
    iv = sorted((max(t.start, lo), min(t.end, hi)) for t in tasks if t.start is not None and t.end is not None and t.end > lo and t.start < hi)
    out = []
    for a, b in iv:
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def gaps(busy, lo, hi):
    g, t = [], lo
    for a, b in busy:
        if a > t:
            g.append((t, a))
        t = max(t, b)
    if t < hi:
        g.append((t, hi))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    p = dict(sysim.PROFILES[a.profile])
    for kv in a.set:
        k, v = kv.split("=", 1)
        p[k] = type(p[k])(float(v)) if not isinstance(p[k], int) else int(float(v))
    sim, marks = sysim.build_and_run(p, a.windows, a.seed)
    res = sysim.analyse(sim, marks, p)
    skip = 3
    lo, hi = marks[skip][0].start, marks[-1][1].end
    nwin = len(marks) - skip
    crit = sorted([t for t in sim.tasks if id(t) in res["path"] and t.end > lo], key=lambda t: t.start)

    def crit_at(a, b):
        """Time of [a, b) by the kind of critical-path task running then."""
        out = {}
        for t in crit:
            x, y = max(a, t.start), min(b, t.end)
            if y > x:
                out[t.cat] = out.get(t.cat, 0.0) + (y - x)
        return out

    groups = {
        "GPU (compute stream)": [t for t in sim.tasks if t.server == "gpu"],
        "CPU expert pool": [t for t in sim.tasks if t.server == "cpu"],
        "host thread": [t for t in sim.tasks if t.server == "host"],
        "NVMe (any read)": [t for t in sim.tasks if t.cat in ("NVMe: demand read", "I/O: read-ahead")],
        "PCIe copy engine": [t for t in sim.tasks if t.server == "copy"],
    }
    win = (hi - lo) / nwin
    print(f"profile {a.profile}: {win * 1e3:.1f} ms a window, {p['accepted'] / win:.1f} tok/s (model)\n")
    for name, tasks in groups.items():
        busy = intervals(tasks, lo, hi)
        idle = gaps(busy, lo, hi)
        tot = sum(b - a for a, b in idle)
        why = {}
        for a0, b0 in idle:
            for k, v in crit_at(a0, b0).items():
                why[k] = why.get(k, 0.0) + v
        print(f"{name}: idle {tot / nwin * 1e3:5.1f} ms a window ({tot / (hi - lo) * 100:4.1f}%) - meanwhile the window was waiting on:")
        for k, v in sorted(why.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {k:28s} {v / nwin * 1e3:5.2f} ms")
    # the GPU's idle stretches by length: short ones only a launch can fill, long ones could hold real work
    gpu_idle = gaps(intervals(groups["GPU (compute stream)"], lo, hi), lo, hi)
    bins = [(0, 50e-6), (50e-6, 200e-6), (200e-6, 500e-6), (500e-6, 1e-3), (1e-3, 1.0)]
    print("\nGPU idle stretches by length (a stretch shorter than a kernel's launch and run cannot be filled):")
    for b0, b1 in bins:
        sel = [b - a for a, b in gpu_idle if b0 <= b - a < b1]
        print(f"  {b0 * 1e6:6.0f}-{b1 * 1e6:6.0f} us: {len(sel) / nwin:5.1f} a window, {sum(sel) / nwin * 1e3:5.2f} ms")


if __name__ == "__main__":
    main()
