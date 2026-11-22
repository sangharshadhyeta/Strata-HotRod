#!/usr/bin/env python3
"""Which eviction policy keeps the best substitutes resident?

Scored on retained gate mass (what the resident-constrained router actually
executes / what it wanted), not hit rate - under substitution, hit rate stops
being the thing worth maximising.

Arms, all using the same sampled eviction (K random candidates, the shape a
real implementation can afford on the hot path - a full scan of 5692 slots
per eviction is not implementable):
  random     evict a sampled candidate arbitrarily - the honesty baseline
  lru        evict the least recently used of the sample
  score      evict the lowest router-score EMA (seeded at admission with the
             score that earned it - without that seed the arm degenerates into
             "evict newest" and the result is meaningless)
  lru+score  even blend of the two
  atlas      evict the resident expert least aligned with the live req_dir
             centroid (decay 0.15, matching production)

Result on Ornith 35B, 4 topic-diverse traces, 1421 slots (18.5%), window 64:

  fill |     policy |  retained |  hit rate
     1 |        lru |    75.72% |    58.63%   <- best on retained
     1 |    random  |    72.51% |    53.31%
     1 |      coact |    72.74% |    58.12%
     1 |      atlas |    71.06% |    51.59%
     1 |      score |    68.90% |    46.27%   <- worst
     4 |        lru |    78.98% |    63.37%
     4 |      coact |    78.61% |    64.88%   <- best on HIT RATE (+1.51pp)
     4 |      atlas |    74.45% |    55.19%

Plain LRU wins on retained gate mass at every fill rate, and every semantic
policy loses to it - atlas and router score lose to random too. Topic
affinity and router score rank experts context-free, with no knowledge of
what was just used, so evicting by them tears up the working set recency is
successfully holding.

Co-activation is the one signal that beats LRU on anything: at fill 4 it
gives +1.51pp HIT RATE, while giving up 0.37pp of retained mass. The two
metrics genuinely diverge - it keeps more of the wanted experts resident,
but the ones LRU keeps are worth more gate mass. It is accumulated ONLINE
here, exactly as production does, so there is no future peeking.

SUPERSEDED - the admission finding below was produced by a model with no
fill cost in it, and is WRONG. See scripts/moe_cost_model.py: admitting an
expert costs throughput, and on real hardware removing the admission gate
buys 10.79pp of hit rate while losing 7.6% of tok/s. With fill cost applied
the ordering inverts and matches the measurement - "admit everything" ranks
last by a wide margin, not first. The eviction rankings above are unaffected,
because those arms all hold the fill rate fixed and so pay the same fill cost.

Admission control, tested separately with LRU eviction, appeared monotonically
harmful on this workload - which is exactly the artifact described above:

  fill | admit_after |  retained |  hit rate
     4 |           1 |    79.00% |    63.36%   admit everything
     4 |           2 |    78.66% |    62.99%   production default, -0.34pp
     4 |           4 |    77.36% |    61.66%

That was the most promising remaining hypothesis, because Belady's edge on a
heavy-tailed stream is refusing to spend a slot on a one-off, and admission
control is how a causal policy does that. It does not transfer.

Standing conclusion: LRU is at or near the practical ceiling for causal
policies here. Belady still beats it by ~17pp, but none of the five signals
available (topic/atlas, router score, co-activation, frequency, admission
demand) approximates reuse distance well enough to close it.
"""
import json, array, random, math

TOPICS = ['code', 'math', 'history', 'medicine']
KEEP, K, DECAY = 64, 8, 0.15


def load(atlas_path='/mnt/nvme/models/ornith/expert-atlas-v2.json'):
    atlas = json.load(open(atlas_path))
    xy = {(c['layer'], c['expert']): (c['x'], c['y']) for c in atlas['cells']}
    ids, scs, lay = [], [], array.array('h')
    for t in TOPICS:
        for line in open(f"traces/deep-{t}.txt"):
            p = line.split(); lay.append(int(p[0]))
            a, b = array.array('h'), array.array('f')
            for x in p[2:2 + KEEP]:
                e, _, s = x.partition(':')
                a.append(int(e)); b.append(float(s) if s else 0.0)
            ids.append(a); scs.append(b)
    return xy, ids, scs, lay


def run(xy, ids, scs, lay, N, fill_rate, window, policy, SAMPLE=32):
    res, keys, pos = {}, [], {}
    rx = ry = 0.0; valid = False; clock = 0
    want_mass = got_mass = 0.0; picks = hits = 0

    def add(k, s0):
        res[k] = [clock, s0]; pos[k] = len(keys); keys.append(k)

    def drop(k):
        i = pos.pop(k); last = keys.pop()
        if i < len(keys):
            keys[i] = last; pos[last] = i
        del res[k]

    for i in range(len(ids)):
        il, a, b = lay[i], ids[i], scs[i]; clock += 1
        want = [(a[j], b[j]) for j in range(K)]
        want_mass += sum(s for _, s in want); picks += K
        for e, _ in want:
            c = xy.get((il, e))
            if c is None:
                continue
            if not valid:
                rx, ry = c; valid = True
            else:
                rx += DECAY * (c[0] - rx); ry += DECAY * (c[1] - ry)
        chosen = []
        for j in range(min(window, len(a))):
            k = (il, a[j])
            if k in res:
                chosen.append(b[j]); r = res[k]
                r[0] = clock; r[1] = 0.8 * r[1] + 0.2 * b[j]
                if len(chosen) == K:
                    break
        got_mass += sum(chosen)
        miss = [(e, s) for e, s in want if (il, e) not in res]
        hits += K - len(miss)
        mag = math.hypot(rx, ry); pushed = 0
        for e, s0 in miss:
            if pushed >= fill_rate:
                break
            k = (il, e); add(k, s0); pushed += 1
            while len(keys) > N:
                cand = random.sample(keys, min(SAMPLE, len(keys)))
                if policy == 'lru':
                    v = min(cand, key=lambda q: res[q][0])
                elif policy == 'score':
                    v = min(cand, key=lambda q: res[q][1])
                elif policy == 'lru+score':
                    v = min(cand, key=lambda q: (res[q][0] / clock) * 0.5 + (res[q][1] / 0.05) * 0.5)
                elif policy == 'random':
                    v = cand[0]
                else:
                    if mag < 1e-9:
                        v = min(cand, key=lambda q: res[q][0])
                    else:
                        def cos(q):
                            c = xy.get(q)
                            if c is None:
                                return -2.0
                            m = math.hypot(*c)
                            return -2.0 if m < 1e-9 else (c[0] * rx + c[1] * ry) / (m * mag)
                        v = min(cand, key=cos)
                if v == k:
                    v = min(cand, key=lambda q: res[q][0])
                drop(v)
    return got_mass / want_mass, hits / picks


if __name__ == '__main__':
    random.seed(11)
    xy, ids, scs, lay = load()
    N = 1421
    print(f"{'fill':>4} | {'policy':>9} | {'retained':>9} | {'hit rate':>9}")
    for fr in (1, 4):
        for pol in ('random', 'lru', 'score', 'lru+score', 'atlas'):
            r, h = run(xy, ids, scs, lay, N, fr, 64, pol)
            print(f"{fr:>4} | {pol:>9} | {r*100:8.2f}% | {h*100:8.2f}%")
