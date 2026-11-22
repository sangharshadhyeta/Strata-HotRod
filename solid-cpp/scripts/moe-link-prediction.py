#!/usr/bin/env python3
"""Label-free test of the discovered atlas: does it predict co-firing it has not seen?

Scoring a discovered atlas by "did it recover our 9 categories" is rigged toward
the probes - it asks whether the structure agrees with our words, not whether it
is real. This asks the question the cache actually cares about: given two experts
in a layer, will they fire in the SAME routing decision later on?

No topic labels anywhere. Traces are pooled, the embedding is built from the
first part of the stream, and the score is AUC over held-out pairs.

Three baselines matter, and the third is the one that keeps this honest:
  probe atlas   cosine between the two experts' measured cats{} vectors
  discovered    cosine in the co-activation embedding
  POPULARITY    product of the two experts' selection frequencies

Popularity is the trap. Frequently-used experts co-fire often simply because
they are frequently used, so a method can post a high AUC while having learned
nothing about which experts belong TOGETHER. Any embedding that fails to beat
popularity has not earned its complexity.
"""
import json, sys, math, random, argparse
from collections import defaultdict
import numpy as np

K_SELECTED = 8   # the router's actual top-k; the trace stores a deeper ranking


def load_pooled(paths, keep=K_SELECTED):
    """All traces, pooled, unlabeled: [(layer, [selected experts])]."""
    seq = []
    for p in paths:
        for line in open(p):
            f = line.split()
            seq.append((int(f[0]), [int(x.split(':')[0]) for x in f[2:2 + keep]]))
    return seq


def cofire_counts(seq):
    c = defaultdict(float)
    freq = defaultdict(int)
    for il, picks in seq:
        for e in picks:
            freq[(il, e)] += 1
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                a, b = picks[i], picks[j]
                if a != b:
                    c[(il, min(a, b), max(a, b))] += 1.0
    return c, freq


def embed_from(counts, dims=16, iters=14, seed=0):
    nodes = {}
    def nid(l, e):
        k = (l, e)
        if k not in nodes:
            nodes[k] = len(nodes)
        return nodes[k]
    r, c, w = [], [], []
    for (il, a, b), v in counts.items():
        r.append(nid(il, a)); c.append(nid(il, b)); w.append(v)
    n = len(nodes)
    r = np.array(r); c = np.array(c); w = np.array(w, dtype=np.float64)
    deg = np.zeros(n); np.add.at(deg, r, w); np.add.at(deg, c, w)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    wn = w * dinv[r] * dinv[c]
    def mv(X):
        Y = np.zeros_like(X)
        np.add.at(Y, r, wn[:, None] * X[c])
        np.add.at(Y, c, wn[:, None] * X[r])
        return Y
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, dims + 4)))
    for _ in range(iters):
        Q, _ = np.linalg.qr(mv(Q))
    B = Q.T @ mv(Q)
    vals, vecs = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(-np.abs(vals))
    return nodes, (Q @ vecs[:, order])[:, :dims], deg


def auc(pos, neg):
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    order = np.argsort(sc)
    ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return (ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_incremental(path):
    """Load a moe-incremental-atlas.py .npz - same (layer,expert)->index keying
    as embed_from's `nodes`, so it drops into the same score() closure below."""
    z = np.load(path, allow_pickle=True)
    nodes = {tuple(int(x) for x in k): int(v) for k, v in zip(z['keys'], z['vals'])}
    emb = z['emb']
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    steps = int(z['steps']) if 'steps' in z.files else -1
    return nodes, emb, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', nargs='+', required=True)
    ap.add_argument('--atlas', default='/mnt/nvme/models/ornith/expert-atlas-v2.json')
    ap.add_argument('--dims', type=int, default=16)
    ap.add_argument('--samples', type=int, default=200000)
    ap.add_argument('--incremental', default=None,
                     help='score a moe-incremental-atlas.py .npz state file too')
    a = ap.parse_args()

    seq = load_pooled(a.traces)
    cut = len(seq) // 2
    train, test = seq[:cut], seq[cut:]
    print(f"pooled decisions: {len(seq):,}  (train {len(train):,} / held-out {len(test):,})")

    tr_counts, tr_freq = cofire_counts(train)
    te_counts, _ = cofire_counts(test)
    print(f"train pairs {len(tr_counts):,}   held-out pairs {len(te_counts):,}")
    w = np.array(list(tr_counts.values()))
    print(f"train edge weight: median {np.median(w):.0f}, mean {w.mean():.1f}, seen-once {np.mean(w<=1)*100:.1f}%")

    nodes, emb, deg = embed_from(tr_counts, dims=a.dims)
    E = emb[:, 1:]                              # drop the degree component
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    atlas = json.load(open(a.atlas)); cats = atlas['categories']
    hv = {}
    for cell in atlas['cells']:
        v = np.zeros(len(cats))
        for k, p in (cell.get('cats') or {}).items():
            if k in cats:
                v[cats.index(k)] = p
        nrm = np.linalg.norm(v)
        if nrm > 0:
            hv[(cell['layer'], cell['expert'])] = v / nrm

    # positives: pairs that co-fire in HELD-OUT data but were never seen in train
    # (predicting a pair you already saw is memorisation, not prediction)
    rng = random.Random(0)
    unseen_pos = [k for k in te_counts if k not in tr_counts]
    rng.shuffle(unseen_pos)
    pos = unseen_pos[:a.samples]
    by_layer = defaultdict(list)
    for (il, e) in nodes:
        by_layer[il].append(e)
    neg = []
    while len(neg) < len(pos):
        il = rng.choice(list(by_layer))
        if len(by_layer[il]) < 2:
            continue
        x, y = rng.sample(by_layer[il], 2)
        k = (il, min(x, y), max(x, y))
        if k not in te_counts and k not in tr_counts:
            neg.append(k)
    print(f"\nheld-out pairs UNSEEN in training: positives {len(pos):,}  negatives {len(neg):,}")

    inc_nodes, E_inc = {}, None
    if a.incremental:
        inc_nodes, E_inc, inc_steps = load_incremental(a.incremental)
        print(f"\nincremental atlas: {a.incremental}  ({len(inc_nodes):,} nodes, {inc_steps:,} total steps)")

    def score(pairs, how):
        out = []
        for (il, x, y) in pairs:
            if how == 'disc':
                i, j = nodes.get((il, x)), nodes.get((il, y))
                out.append(float(E[i] @ E[j]) if i is not None and j is not None else 0.0)
            elif how == 'incr':
                i, j = inc_nodes.get((il, x)), inc_nodes.get((il, y))
                out.append(float(E_inc[i] @ E_inc[j]) if i is not None and j is not None else 0.0)
            elif how == 'probe':
                u, v = hv.get((il, x)), hv.get((il, y))
                out.append(float(u @ v) if u is not None and v is not None else 0.0)
            else:
                out.append(math.log1p(tr_freq.get((il, x), 0)) + math.log1p(tr_freq.get((il, y), 0)))
        return np.array(out)

    print(f"\n{'method':<28} {'AUC':>7}   (0.500 = no better than chance)")
    methods = [("popularity (control)", 'pop'),
               ("probe atlas (9 cats)", 'probe'),
               ("DISCOVERED co-activation (spectral)", 'disc')]
    if a.incremental:
        methods.append(("INCREMENTAL co-activation (SGNS)", 'incr'))
    for label, how in methods:
        print(f"  {label:<26} {auc(score(pos, how), score(neg, how)):.4f}")


if __name__ == '__main__':
    main()
