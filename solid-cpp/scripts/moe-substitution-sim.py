#!/usr/bin/env python3
"""Offline cost model for resident-constrained MoE routing.

Answers one question: if an expert the router wanted is not in VRAM, what does
it cost to run the best RESIDENT expert instead of stalling to fetch it?

Needs traces from llama-moe-trace with MOE_TRACE_SCORES=1 and a deep MOE_TOPK
(128 used here). Top-8 alone is not enough - the substitute is found further
down the ranking, so the tail is the whole point.

Two knobs, and the comparison between them is the result:
  window    how many ranked candidates the substitution may consider. 8 means
            no substitution at all (missed picks are simply dropped); 16 is
            what a decision made ON the dispatch path can afford; 64 is what
            becomes affordable once the residency mask is built EARLY, on the
            CPU, off the critical path (the Track 1.5 mechanism).
  fill_rate how many experts may be pushed into VRAM per decision - the PCIe
            budget, i.e. how hard the system "races" to fill.

Admission always follows what the router wanted UNCONSTRAINED, never what it
was forced to execute. That is deliberate: feeding admission from realized
selections would close a self-confirming loop, where resident experts are the
only ones ever picked and so the only ones that ever stay resident.
"""
import collections, array, sys

TOPICS = ['code', 'math', 'history', 'medicine']
KEEP, K = 64, 8


def load(prefix='traces/deep-'):
    ids, scs, lay = [], [], array.array('h')
    for t in TOPICS:
        for line in open(f"{prefix}{t}.txt"):
            p = line.split()
            lay.append(int(p[0]))
            a, b = array.array('h'), array.array('f')
            for x in p[2:2 + KEEP]:
                e, _, s = x.partition(':')
                a.append(int(e)); b.append(float(s) if s else 0.0)
            ids.append(a); scs.append(b)
    return ids, scs, lay


def run(ids, scs, lay, N, fill_rate, window):
    cache = collections.OrderedDict()
    want_mass = got_mass = 0.0
    subs = picks = 0
    for i in range(len(ids)):
        il, a, b = lay[i], ids[i], scs[i]
        want = [(a[j], b[j]) for j in range(K)]
        want_mass += sum(s for _, s in want); picks += K
        chosen = []
        for j in range(min(window, len(a))):
            if (il, a[j]) in cache:
                chosen.append((a[j], b[j]))
                if len(chosen) == K:
                    break
        got_mass += sum(s for _, s in chosen)
        subs += sum(1 for e, _ in want if (il, e) not in cache)
        for e, _ in chosen:
            cache.move_to_end((il, e))
        pushed = 0
        for e, _ in want:
            if pushed >= fill_rate:
                break
            if (il, e) not in cache:
                cache[(il, e)] = True; pushed += 1
                if len(cache) > N:
                    cache.popitem(last=False)
    return got_mass / want_mass, subs / picks


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1421
    ids, scs, lay = load()
    print(f"decisions: {len(ids)}   cache: {N} slots")
    print(f"{'fill/dec':>8} | {'window':>6} | {'retained':>9} | {'substituted':>11}")
    for fr in (1, 2, 4, 8):
        for w in (8, 16, 64):
            r, s = run(ids, scs, lay, N, fr, w)
            print(f"{fr:>8} | {w:>6} | {r*100:8.2f}% | {s*100:10.2f}%")
