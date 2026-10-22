"""A model of the decode's read-ahead and the drive, to compare the read-ahead settings (STRATA_IO_PF_THREADS, STRATA_IO_EDF,
STRATA_IO_PF_YIELD, STRATA_LA_MODEL) before a run on the GPU.  It is NOT the engine: no kernels run.  It replays what
expert_source.cpp does with a layer's guesses and demand reads against a drive that shares its bandwidth among the reads in flight.

Per decode window, per MoE layer L:
  1. the GPU reaches L (doorbell); the host queues the look-ahead's guesses for L+1 (dist 1) and L+2 (dist 2)
  2. the host needs L's cold rank 0-1 experts (not in VRAM, not in RAM): a landed guess costs nothing, one in flight is waited for,
     a queued or unpredicted one is read by the host now (a demand read)
  3. the CPU computes the layer's misses, the GPU finishes the layer and reaches L+1
The drive: B bytes/s shared equally by every read in flight (O_DIRECT, one blob per read).  Guesses run on `workers` threads.

Parameters default to the 19-21 Oct measurements in docs/perf-ideas.md (RTX 3060 box, PCIe 3.0 NVMe 3.3 GB/s, IQ3_S 2.33 MB experts)
and tools/predictor (top-2 recall of rank 0-1: router reuse 55.3% at dist 1, 46.5% at dist 2; the published model 67.5-71.7% at dist 1).
usage: python sim_readahead.py [--cold 0.27] [--windows 400] [--seed 1]"""
import argparse
import random

BLOB = 2.33e6


class Drive:
    def __init__(s, bw, workers, edf, yield_):
        s.bw, s.workers, s.edf, s.yield_ = bw, workers, edf, yield_
        s.t = 0.0
        s.active = {}          # key -> [remaining bytes, is_guess]
        s.staged = set()       # keys whose read has finished
        s.queue = []           # guesses: [layer, key, window, from_layer]
        s.cur = (-1, -1)       # (window, layer) the host routes now
        s.demand_active = 0
        s.read_guess = s.read_demand = 0
        s.stale = 0

    def guesses_in_flight(s):
        return sum(1 for v in s.active.values() if v[1])

    def try_start(s):
        while s.queue:
            if s.guesses_in_flight() >= s.workers or (s.yield_ and s.demand_active):
                return
            w, l = s.cur
            if s.edf:
                before = len(s.queue)
                s.queue = [q for q in s.queue if q[2] == w and q[0] > l]
                s.stale += before - len(s.queue)
                if not s.queue:
                    return
                i = min(range(len(s.queue)), key=lambda j: (s.queue[j][0], -s.queue[j][3], j))
            else:
                i = len(s.queue) - 1   # the newest first
            q = s.queue.pop(i)
            if q[1] in s.staged or q[1] in s.active:
                continue
            s.active[q[1]] = [BLOB, True]
            s.read_guess += 1

    def advance(s, until=None, done=None):
        """Runs the drive to time `until`, or until done() holds (checked at each completion)."""
        while True:
            if done is not None and done():
                return
            n = len(s.active)
            if n == 0:
                if until is None:
                    raise RuntimeError("waiting on nothing")
                s.t = max(s.t, until)
                return
            rate = s.bw / n
            dt = min(v[0] for v in s.active.values()) / rate
            if until is not None and s.t + dt >= until:
                for v in s.active.values():
                    v[0] -= (until - s.t) * rate
                s.t = until
                return
            s.t += dt
            fin = []
            for k, v in s.active.items():
                v[0] -= dt * rate
                if v[0] <= 1.0:
                    fin.append(k)
            for k in fin:
                del s.active[k]
                s.staged.add(k)
            s.try_start()

    def demand(s, keys):
        """The host reads what it needs and has not got; returns the counts by what the read-ahead had done."""
        got = {"landed": 0, "in_flight": 0, "queued": 0, "unpredicted": 0}
        need = []
        queued = {q[1] for q in s.queue}
        for k in keys:
            if k in s.staged:
                got["landed"] += 1
            elif k in s.active:
                got["in_flight"] += 1
                need.append(k)
            else:
                got["queued" if k in queued else "unpredicted"] += 1
                s.queue = [q for q in s.queue if q[1] != k]
                s.active[k] = [BLOB, False]
                s.read_demand += 1
                need.append(k)
        if need:
            s.demand_active += 1
            s.advance(done=lambda: all(k in s.staged for k in need))
            s.demand_active -= 1
            s.try_start()
        return got


def run(a, threads=None, edf=False, yield_=False, r1=0.553, r2=0.465, wrong_scale=1.0, label=""):
    rng = random.Random(a.seed)
    d = Drive(a.bw, threads or a.workers, edf, yield_)   # STRATA_IO_PF_THREADS: one guess read per thread
    # wrong guesses per layer and dist, sized so that the baseline uses ~35% of what it reads ahead (measured 32-41%)
    wrong1 = a.cold * a.wrong * wrong_scale
    wrong2 = a.cold * a.wrong
    wait_total = cpu_total = 0.0
    counts = {"landed": 0, "in_flight": 0, "queued": 0, "unpredicted": 0}
    t0 = 0.0

    def pois(lam):
        n, p, x = 0, 1.0, rng.random()
        import math
        e = math.exp(-lam)
        while True:
            p *= rng.random() if n else x
            if p < e:
                return n
            n += 1

    for w in range(a.windows):
        miss = [[(w, L, e) for e in rng.sample(range(512), pois(a.cold))] for L in range(48)]
        pred1 = [[k for k in miss[L] if rng.random() < r1] for L in range(48)]
        pred2 = [[k for k in pred1[L] if rng.random() < r2 / r1] for L in range(48)]   # a dist-2 hit is usually a dist-1 hit too
        for L in range(48):
            d.advance(until=d.t)
            d.cur = (w, L)
            for dist, preds, wr in ((1, pred1, wrong1), (2, pred2, wrong2)):
                T = L + dist
                if T >= 48:
                    continue
                g = list(preds[T]) + [(w, T, 1000 + rng.randrange(10**6)) for _ in range(pois(wr))]
                rng.shuffle(g)
                for k in g:
                    d.queue.append([T, k, w, L])
            d.try_start()
            ts = d.t
            got = d.demand(miss[L])
            for k in counts:
                counts[k] += got[k]
            wait_total += d.t - ts
            cpu = a.cpu_layer + a.cpu_miss * len(miss[L])
            cpu_total += cpu
            d.advance(until=d.t + cpu + a.gpu_layer)
        d.advance(until=d.t + a.window_extra)    # commit, draft, sampling: the drive keeps reading
    windows_ms = (d.t - t0) / a.windows * 1e3
    n_miss = sum(counts.values())
    used = counts["landed"] + counts["in_flight"]
    return dict(label=label, window_ms=windows_ms, wait_ms=wait_total / a.windows * 1e3, tok_s=a.tok_window / (windows_ms / 1e3),
                landed=counts["landed"] / n_miss, in_flight=counts["in_flight"] / n_miss, queued=counts["queued"] / n_miss,
                unpred=counts["unpredicted"] / n_miss, used_share=used / max(1, d.read_guess), mb_window=(d.read_guess + d.read_demand) * BLOB / 1e6 / a.windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", type=float, default=0.27, help="cold rank 0-1 experts (from the drive) per layer of a window (~13 a window)")
    ap.add_argument("--wrong", type=float, default=1.1, help="wrong guesses per right one per dist (sets the read-ahead's used share)")
    ap.add_argument("--bw", type=float, default=3.3e9)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu-layer", dest="gpu_layer", type=float, default=0.52e-3, help="GPU time per layer after the host's rows (25 ms / 48)")
    ap.add_argument("--cpu-layer", dest="cpu_layer", type=float, default=0.15e-3)
    ap.add_argument("--cpu-miss", dest="cpu_miss", type=float, default=0.25e-3)
    ap.add_argument("--window-extra", dest="window_extra", type=float, default=4e-3)
    ap.add_argument("--tok-window", dest="tok_window", type=float, default=3.3)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    model = dict(r1=0.69, wrong_scale=(1 - 0.69) / (1 - 0.553))   # the published predictor at dist 1 (67.5-71.7%); fewer wrong guesses
    arms = [
        run(a, label="today (8 threads, newest first)"),
        run(a, edf=True, label="EDF"),
        run(a, threads=2, label="PF_THREADS=2"),
        run(a, threads=1, label="PF_THREADS=1"),
        run(a, threads=1, edf=True, label="PF_THREADS=1 + EDF"),
        run(a, threads=1, yield_=True, label="PF_THREADS=1 + YIELD"),
        run(a, threads=1, edf=True, yield_=True, label="PF_THREADS=1 + EDF + YIELD"),
        run(a, **model, label="LA_MODEL alone"),
        run(a, threads=1, yield_=True, **model, label="PF_THREADS=1 + YIELD + LA_MODEL"),
        run(a, threads=1, edf=True, yield_=True, **model, label="PF_THREADS=1 + EDF + YIELD + LA_MODEL"),
        run(a, r1=1.0, r2=1.0, wrong_scale=0.0, label="(bound: every miss predicted, no wrong guess)"),
    ]
    base = arms[0]
    print(f"cold rank 0-1 misses {a.cold:.2f} a layer ({a.cold * 48:.0f} a window), drive {a.bw / 1e9:.1f} GB/s, {a.windows} windows\n")
    print(f"{'arm':44s} {'tok/s':>6} {'vs today':>8} {'window':>7} {'read wait':>9}  {'landed':>6} {'reading':>7} {'queued':>6} {'unpred':>6}  {'RA used':>7} {'MB/win':>6}")
    for r in arms:
        print(f"{r['label']:44s} {r['tok_s']:6.1f} {(r['tok_s'] / base['tok_s'] - 1) * 100:+7.1f}% {r['window_ms']:6.1f}ms {r['wait_ms']:7.2f}ms"
              f"  {r['landed'] * 100:5.0f}% {r['in_flight'] * 100:6.0f}% {r['queued'] * 100:5.0f}% {r['unpred'] * 100:5.0f}%"
              f"  {r['used_share'] * 100:6.0f}% {r['mb_window']:6.0f}")


if __name__ == "__main__":
    main()
