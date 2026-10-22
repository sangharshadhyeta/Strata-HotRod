"""Offline check of idle-drive RAM-tier promotions on a STRATA_ROUTER_TRACE file (RTR1, see trace_reader.py).

Question (docs/perf-ideas.md, 22 Oct, "Idle time as waste"): if the drive's idle time in each decode window promoted experts into the
RAM tier, chosen by a recency or frequency rule, what share of those promotions would a later window have read cold - and how many
cold reads would the decode save net of what the promotions evict?

Replay, window by window (a trace `step` is one decode window), per MoE layer:
  * the decode NEEDS the distinct experts its tokens pick at ranks 0..need_ranks-1 (rank 0-1 exact; ranks 2-9 take stand-ins)
  * VRAM tier: per layer the `vram` experts with the highest decayed routing count (all ten ranks, as the card's swap rounds see
    them), refreshed every window within a swap budget - an approximation of the adaptive tier
  * RAM tier: `ram` experts in all, first filled with the most-routed experts of the first `warm_share` of the trace (the profile-ranked
    fixed tier); an expert needed and in neither tier is a COLD read (the drive)
  * promotions: after each window, up to `budget` experts not in VRAM or RAM are read into the RAM tier (what the drive's idle time
    holds: ~15-22 a window on the RTX 3060 box), chosen by the rule; when the tier is full each one evicts the RAM expert with the
    lowest decayed count.  They land `lag` windows later.
Reported per rule: promotions a window, the USEFUL share (a promoted expert needed while still promoted, before eviction - the sysim
`fill_useful`), cold reads saved and cold reads caused by evictions (a later need of an evicted expert), and the net change.

usage: python promote_check.py trace1.bin [--hidden 2560] [--vram 58] [--ram 4400] [--budget 15] [--max-windows N]
       python promote_check.py --synthetic out.bin   (writes a small test trace with locality, hidden size 8)
"""
import argparse
import heapq
import mmap
import random
import struct
import sys
from collections import defaultdict

RULES = ("recency", "frequency", "recency+frequency")


def read_windows(path, hidden, need_ranks, max_windows=None):
    """Yields (step, {layer: (need set, all-ranks list)}) per decode window; the bf16 inputs are skipped, never loaded."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:4] != b"RTR1":
            sys.exit("not an RTR1 trace")
        off, size = 4, len(mm)
        cur, layers, n = None, {}, 0
        while off + 8 <= size:
            step, layer, nt, k = struct.unpack_from("<IHBB", mm, off)
            rec = 8 + nt * (hidden * 2 + k * 2)
            if off + rec > size:
                break
            if cur is not None and step != cur:
                yield cur, layers
                n += 1
                if max_windows and n >= max_windows:
                    return
                layers = {}
            cur = step
            p = off + 8
            need, every = set(), []
            for _ in range(nt):
                p += hidden * 2
                ids = struct.unpack_from(f"<{k}h", mm, p)
                p += k * 2
                need.update(e for e in ids[:need_ranks] if e >= 0)
                every.extend(e for e in ids if e >= 0)
            layers[layer] = (need, every)
            off += rec
        if cur is not None and layers:
            yield cur, layers


def replay(windows, a, rule):
    decay = a.decay
    inc = [1.0]
    cnt = defaultdict(float)                 # (layer, expert) -> decayed count of ALL ranks (the card's view)
    need_cnt = defaultdict(float)            # (layer, expert) -> decayed count of needed picks (rank 0..need-1)
    last_need = {}                           # (layer, expert) -> window index of its last need
    vram = defaultdict(set)
    n_warm = max(1, int(len(windows) * a.warm_share))
    tally = defaultdict(int)
    for _, layers in windows[:n_warm]:
        for l, (need, every) in layers.items():
            for e in every:
                tally[(l, e)] += 1
    # the card and the RAM tier start as the profile ranks them: per layer its most-routed experts on the card, the next ones in RAM
    per_layer = defaultdict(list)
    for (l, e), c in tally.items():
        per_layer[l].append((c, e))
    for l, lst in per_layer.items():
        lst.sort(reverse=True)
        vram[l] = set(e for _, e in lst[:a.vram])
    ram = set(k for k, _ in sorted(((k, c) for k, c in tally.items() if k[1] not in vram[k[0]]), key=lambda kv: -kv[1])[:a.ram])
    promoted = {}                            # key -> window it landed (promoted and still in RAM)
    pending = []                             # (land window, key)
    st = dict(windows=0, cold=0, promotions=0, useful=0, saved=0, caused=0)
    used_once = set()                        # (key, landing window): a promotion counted useful once
    evicted_by_promo = {}                    # key -> window evicted (by a promotion), for "caused" accounting
    for wi, (_, layers) in enumerate(windows):
        while pending and pending[0][0] <= wi:   # promotions landing now
            _, key = heapq.heappop(pending)
            if key in ram or key in vram[key[0]]:
                continue
            if len(ram) >= a.ram:            # evict the lowest decayed count (never the one landing)
                victim = min(ram, key=lambda k: (cnt[k], need_cnt[k]))
                ram.discard(victim)
                promoted.pop(victim, None)
                evicted_by_promo[victim] = wi
            ram.add(key)
            promoted[key] = wi
            st["promotions"] += 1
        if wi >= n_warm:
            st["windows"] += 1
        for l, (need, every) in layers.items():
            for e in need:
                key = (l, e)
                if e in vram[l]:
                    continue
                if key in ram:
                    if key in promoted and wi >= n_warm:
                        st["saved"] += 1
                        if (key, promoted[key]) not in used_once:
                            used_once.add((key, promoted[key]))
                            st["useful"] += 1
                    continue
                if wi >= n_warm:
                    st["cold"] += 1
                    if key in evicted_by_promo and wi - evicted_by_promo[key] <= a.horizon:
                        st["caused"] += 1
        # counts (decayed lazily: an increment that grows by 1/decay a window), then the VRAM tier's swap round
        inc[0] /= decay
        if inc[0] > 1e100:
            for d in (cnt, need_cnt):
                for k in d:
                    d[k] /= inc[0]
            inc[0] = 1.0
        for l, (need, every) in layers.items():
            for e in every:
                cnt[(l, e)] += inc[0]
            for e in need:
                need_cnt[(l, e)] += inc[0]
                last_need[(l, e)] = wi
        swaps = a.swaps
        floor = 2.0 * inc[0] * decay          # "a decayed count of 2 or more", in this window's units
        cand = sorted(((cnt[k], k) for k in cnt if k[1] not in vram[k[0]] and cnt[k] >= floor), reverse=True)
        for c, (l, e) in cand:
            if swaps <= 0:
                break
            if len(vram[l]) >= a.vram:
                victim = min(vram[l], key=lambda x: cnt[(l, x)])
                if cnt[(l, victim)] * a.min_gain >= c:
                    continue
                vram[l].discard(victim)
            vram[l].add(e)
            swaps -= 1
        # promotions for the drive's idle time in the next window: candidates not in VRAM or RAM
        if rule == "recency":
            score = {k: -(wi - w) for k, w in last_need.items() if wi - w <= a.recent}
        elif rule == "frequency":
            score = {k: v for k, v in need_cnt.items() if v >= 0.5 * inc[0]}
        else:   # recency+frequency: the decayed need count, only experts needed recently
            score = {k: need_cnt[k] for k, w in last_need.items() if wi - w <= a.recent}
        cands = [k for k in sorted(score, key=lambda k: -score[k]) if k not in ram and k[1] not in vram[k[0]]]
        queued = {k for _, k in pending}
        for k in cands[:a.budget]:
            if k not in queued:
                heapq.heappush(pending, (wi + a.lag, k))
    n_prom_eval = max(1, st["promotions"])
    w = max(1, st["windows"])
    return dict(rule=rule, prom_w=st["promotions"] / max(1, len(windows)), useful=st["useful"] / n_prom_eval,
                saved_w=st["saved"] / w, caused_w=st["caused"] / w, cold_w=st["cold"] / w)


def baseline(windows, a):
    q = argparse.Namespace(**vars(a))
    q.budget = 0
    return replay(windows, q, "frequency")["cold_w"]


def synthetic(path, n_windows=1500, layers=48, experts=512, k=10, hidden=8, seed=7):
    """Routing with locality: each layer has a slowly drifting topic set (Zipf inside it) plus noise; 3 tokens a window."""
    rng = random.Random(seed)
    topics = [[rng.sample(range(experts), 40) for _ in range(6)] for _ in range(layers)]
    with open(path, "wb") as f:
        f.write(b"RTR1")
        topic = 0
        for w in range(n_windows):
            if rng.random() < 0.01:
                topic = rng.randrange(6)
            nt = 3
            for l in range(layers):
                f.write(struct.pack("<IHBB", w, l, nt, k))
                for _ in range(nt):
                    f.write(b"\x00" * hidden * 2)
                    ids = []
                    while len(ids) < k:
                        e = topics[l][topic][min(39, int(rng.paretovariate(1.2)) - 1)] if rng.random() < 0.8 else rng.randrange(experts)
                        if e not in ids:
                            ids.append(e)
                    f.write(struct.pack(f"<{k}h", *ids))
    print(f"wrote {path}: {n_windows} windows, {layers} layers, hidden {hidden}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--synthetic", metavar="OUT")
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--need-ranks", dest="need_ranks", type=int, default=2, help="ranks the decode must compute exactly (2: ranks 0-1)")
    ap.add_argument("--vram", type=int, default=58, help="VRAM experts per layer (2,793 on the 3060 card / 48)")
    ap.add_argument("--ram", type=int, default=4400, help="RAM tier size in experts (10 GiB / 2.33 MB)")
    ap.add_argument("--warm-share", dest="warm_share", type=float, default=0.2, help="share of the trace that ranks the fixed RAM tier")
    ap.add_argument("--budget", type=int, nargs="*", default=[5, 10, 15, 22], help="promotions a window (the drive's idle time)")
    ap.add_argument("--lag", type=int, default=1, help="windows until a promotion lands")
    ap.add_argument("--decay", type=float, default=0.7, help="the adaptive tier's decay a window")
    ap.add_argument("--swaps", type=int, default=12, help="VRAM swaps a window")
    ap.add_argument("--min-gain", dest="min_gain", type=float, default=1.5)
    ap.add_argument("--recent", type=int, default=8, help="recency window (windows) for the recency rules")
    ap.add_argument("--horizon", type=int, default=64, help="a cold read within this many windows of an eviction counts as caused by it")
    ap.add_argument("--max-windows", dest="max_windows", type=int)
    a = ap.parse_args()
    if a.synthetic:
        synthetic(a.synthetic)
        return
    if not a.trace:
        ap.error("a trace file, or --synthetic OUT")
    windows = list(read_windows(a.trace, a.hidden, a.need_ranks, a.max_windows))
    budgets = a.budget
    print(f"{a.trace}: {len(windows)} decode windows; VRAM {a.vram} a layer, RAM tier {a.ram}, needed ranks 0-{a.need_ranks - 1}; "
          f"the first {a.warm_share:.0%} ranks the fixed tier and is not scored\n")
    base = baseline(windows, a)
    print(f"no promotions: {base:.2f} cold reads a window\n")
    print(f"{'rule':18s} {'budget':>6} {'promoted/win':>12} {'useful':>7} {'saved/win':>9} {'caused/win':>10} {'net cold/win':>12} {'change':>7}")
    best = None
    for rule in RULES:
        for b in budgets:
            a.budget = b
            r = replay(windows, a, rule)
            net = r["cold_w"]
            print(f"{rule:18s} {b:6d} {r['prom_w']:12.1f} {r['useful'] * 100:6.1f}% {r['saved_w']:9.2f} {r['caused_w']:10.2f} {net:12.2f} "
                  f"{(net / base - 1) * 100 if base else 0:+6.1f}%")
            if best is None or net < best[2]:
                best = (rule, b, net, r["useful"])
    if best and base:
        print(f"\nbest: {best[0]}, {best[1]} a window: cold reads {base:.2f} -> {best[2]:.2f} a window, {best[3] * 100:.0f}% of promotions useful")
        print(f"into the simulator: python tools/sysim/sysim.py --set idle_fill=1 fill_useful={best[3]:.2f} fill_chunk=290000 "
              f"(or, as fewer cold misses: cold_share={0.2 * best[2] / base:.3f})")


if __name__ == "__main__":
    main()
