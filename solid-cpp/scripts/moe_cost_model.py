#!/usr/bin/env python3
"""Fill-cost model, calibrated against real measurements on 8099.

Both policy simulators originally scored on hits and retained gate mass alone,
with no notion that ADMITTING an expert costs anything. That blind spot is not
academic: it made them recommend "admit everything", which on real hardware is
the worst configuration tested - it wins 10.79pp of hit rate and loses 7.6% of
throughput, because the fill traffic costs more than the hits save.

The cache already carries a refill-cost notion of its own
(MOE_CACHE_COST_TIER_NVME = 5.8 vs RAM = 1.0 in moe-cache.cu), but that weight
is about WHICH expert is dearer to refetch - page-cache-resident or gone to
NVMe. It says nothing about the throughput cost of doing a fill at all, which
is the term the simulators were missing.

Rather than invent constants, this fits tok/s to three measured arms, all on
Ornith 35B / RTX 3060, same 4-prompt protocol, warmed to steady state:

  arm                        tok/s   hit      fills(evictions)
  baseline(admit2,readmit8)  56.75   68.46%    23,714
  ADMIT_AFTER=1              57.36   68.81%    23,405
  ADMIT=1+THROTTLE=1         52.43   79.25%   110,169

Picks per measurement window is 584,640, from the identical-protocol sweep
run earlier, and is what converts fill counts into a per-pick rate.

Caveats, because this is a calibration and not a law:
  - three points, one workload, one model, one GPU
  - linear in both terms; the real fill cost saturates once PCIe is the
    bottleneck, so it will overstate the cost of small fill counts and
    understate it near saturation
  - the simulators count (layer, expert) units while the server counts
    per-tensor fills; the ratio between them is assumed constant
Use it to RANK policies, not to predict absolute throughput.
"""

PICKS_PER_WINDOW = 584_640

ARMS = [
    # (tok/s, hit fraction, fills)
    (56.75, 0.6846,  23_714),
    (57.36, 0.6881,  23_405),
    (52.43, 0.7925, 110_169),
]


def _solve():
    """tok/s = a + b*hit + c*fill_ratio   (c comes out negative)"""
    rows = [[1.0, h, f / PICKS_PER_WINDOW, t] for (t, h, f) in ARMS]
    n = 3
    for i in range(n):                       # gaussian elimination, 3x3
        p = max(range(i, n), key=lambda r: abs(rows[r][i]))
        rows[i], rows[p] = rows[p], rows[i]
        d = rows[i][i]
        rows[i] = [v / d for v in rows[i]]
        for r in range(n):
            if r != i and rows[r][i]:
                m = rows[r][i]
                rows[r] = [v - m * w for v, w in zip(rows[r], rows[i])]
    return rows[0][3], rows[1][3], rows[2][3]


A, B, C = _solve()


def predicted_tps(hit_fraction, fills, picks):
    """Rank-order estimate of throughput for a policy's hit rate and fill count."""
    return A + B * hit_fraction + C * (fills / picks)


def fill_cost_tps(fills, picks):
    """Throughput given up to perform this many fills."""
    return -C * (fills / picks)


if __name__ == '__main__':
    print(f"tok/s = {A:.3f} + {B:.3f}*hit + {C:.3f}*fills_per_pick")
    print(f"  one point of hit rate is worth : {B/100:.4f} tok/s")
    print(f"  one fill per 1000 picks costs  : {-C/1000:.4f} tok/s")
    be = (B / 100) / (-C / 1000)
    print(f"  break-even: a policy must gain 1pp of hit rate per {be:.1f} extra fills/1000 picks")
    print("\n  refit check against the measured arms:")
    for t, h, f in ARMS:
        print(f"    measured {t:6.2f}  predicted {predicted_tps(h, f, PICKS_PER_WINDOW):6.2f}")
