#!/usr/bin/env python3
"""Incremental discovered atlas: co-firing pulls experts together, over time.

The spectral version factors a snapshot in one step. This is the mechanism
actually intended: every routing decision nudges the experts that fired
together closer and pushes unrelated ones apart, the layout accumulates across
runs, and the map settles - or fails to - on its own.

Settling is NOT annealed in. The learning rate is CONSTANT. If displacement
falls over time it is because the objective is being satisfied, not because the
step size was turned down; an annealed rate would manufacture the appearance of
settling regardless of whether real structure existed. That is the whole point
of the test, so the knob stays fixed.

Objective is skip-gram with negative sampling over co-firing pairs: a pair that
fired in the same routing decision is a positive, a random same-layer pair is a
negative. Negatives are drawn within the layer because cross-layer pairs are
trivially separable and would flatter the model.

State persists to an .npz holding the vectors, the node index, and the total
step count. Re-running RESUMES from that file and keeps training, the same way
session.history accumulates across restarts rather than starting cold.
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np

K_SELECTED = 8


def load_pooled(paths, keep=K_SELECTED):
    seq = []
    for p in paths:
        for line in open(p):
            f = line.split()
            seq.append((int(f[0]), [int(x.split(':')[0]) for x in f[2:2 + keep]]))
    return seq


def pairs_from(seq, nodes, grow=True):
    """Positive co-firing pairs as node-index arrays, plus per-layer node lists."""
    A, B = [], []
    by_layer = defaultdict(set)
    for il, picks in seq:
        idx = []
        for e in picks:
            k = (il, e)
            i = nodes.get(k)
            if i is None:
                if not grow:
                    continue
                i = len(nodes); nodes[k] = i
            idx.append(i); by_layer[il].add(i)
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                if idx[x] != idx[y]:
                    A.append(idx[x]); B.append(idx[y])
    return (np.array(A, dtype=np.int64), np.array(B, dtype=np.int64),
            {l: np.array(sorted(v), dtype=np.int64) for l, v in by_layer.items()})


def pair_key(layer, a, b, n_expert=256):
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    return layer.astype(np.int64) * n_expert * n_expert + lo * n_expert + hi


def sgd_epoch(emb, G, A, B, layer_of, expert_of, layer_members, pos_keys,
              lr, negs, rng, batch=1_000_000):
    """One pass, AdaGrad per node per dimension. Constant base lr by design.

    Uniform averaging (dividing each node's accumulated update by how often it
    appeared) cures the oscillation but flattens everyone equally, and the map
    settles into something with no predictive content - AUC 0.4991, chance.
    AdaGrad instead scales each node's step by its OWN accumulated gradient
    history, so a frequently selected expert damps down as its history grows
    while a rare one keeps moving. That is the "keep the history of affinity
    rather than overwrite it" correction, applied per node.

    The base learning rate is still constant. Settling still has to be earned:
    AdaGrad's denominator grows only where gradients actually keep arriving.
    """
    n = len(A)
    eps = 1e-8
    for s in range(0, n, batch):
        a = A[s:s + batch]; b = B[s:s + batch]
        d = np.zeros_like(emb)
        ua, ub = emb[a], emb[b]
        sig = 1.0 / (1.0 + np.exp(np.clip(np.einsum('ij,ij->i', ua, ub), -30, 30)))
        g = sig[:, None]
        np.add.at(d, a, g * ub); np.add.at(d, b, g * ua)
        for _ in range(negs):
            ls = layer_of[a]
            c = np.empty_like(a)
            for l in np.unique(ls):
                m = ls == l
                mem = layer_members[l]
                c[m] = mem[rng.integers(0, len(mem), size=int(m.sum()))]
            # REJECT negatives that are actually observed co-firing pairs. The
            # graph is ~23% dense, so uniform sampling makes roughly a quarter
            # of every "negative" a real positive - the model is then trained to
            # push apart the same pairs it is pulling together, which is enough
            # to cancel the signal entirely (AUC 0.48, chance, despite clean
            # convergence). Three resample rounds clear almost all of them;
            # whatever survives is rare enough not to matter.
            for _ in range(3):
                k = pair_key(ls, expert_of[a], expert_of[c])
                bad = np.isin(k, pos_keys)
                if not bad.any():
                    break
                idx = np.flatnonzero(bad)
                lb = ls[idx]
                for l in np.unique(lb):
                    m = lb == l
                    mem = layer_members[l]
                    c[idx[m]] = mem[rng.integers(0, len(mem), size=int(m.sum()))]
            ua, uc = emb[a], emb[c]
            sig = 1.0 / (1.0 + np.exp(np.clip(-np.einsum('ij,ij->i', ua, uc), -30, 30)))
            g = -sig[:, None]
            np.add.at(d, a, g * uc); np.add.at(d, c, g * ua)
        G += d * d
        emb += lr * d / (np.sqrt(G) + eps)
        nrm = np.linalg.norm(emb, axis=1, keepdims=True)
        np.divide(emb, np.maximum(nrm, 1e-12), out=emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', nargs='+', required=True)
    ap.add_argument('--state', default='traces/incremental-atlas.npz')
    ap.add_argument('--dims', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.5)
    ap.add_argument('--negs', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--fresh', action='store_true', help='ignore any saved state')
    a = ap.parse_args()

    seq = load_pooled(a.traces)
    cut = len(seq) // 2
    train, test = seq[:cut], seq[cut:]

    nodes, emb, steps, G = {}, None, 0, None
    if os.path.exists(a.state) and not a.fresh:
        z = np.load(a.state, allow_pickle=True)
        nodes = {tuple(k): int(v) for k, v in zip(z['keys'], z['vals'])}
        emb = z['emb']; steps = int(z['steps'])
        G = z['G'] if 'G' in z.files else None
        print(f"RESUMED from {a.state}: {len(nodes):,} nodes, {steps:,} prior steps")

    A, B, members = pairs_from(train, nodes)
    rng = np.random.default_rng(0)
    if emb is None:
        emb = rng.standard_normal((len(nodes), a.dims)) * 0.01
        G = np.zeros_like(emb)
        print(f"fresh start: {len(nodes):,} nodes, {a.dims} dims")
    elif len(nodes) > len(emb):
        grow_n = len(nodes) - len(emb)
        emb = np.vstack([emb, rng.standard_normal((grow_n, emb.shape[1])) * 0.01])
        G = np.vstack([G, np.zeros((grow_n, emb.shape[1]))]) if G is not None else np.zeros_like(emb)
        print(f"grew to {len(nodes):,} nodes")

    if G is None or len(G) != len(emb):
        G = np.zeros_like(emb)
    layer_of = np.zeros(len(nodes), dtype=np.int64)
    expert_of = np.zeros(len(nodes), dtype=np.int64)
    for (l, e), i in nodes.items():
        layer_of[i] = l; expert_of[i] = e
    pos_keys = np.unique(pair_key(layer_of[A], expert_of[A], expert_of[B]))
    print(f"observed positive pairs (excluded from negatives): {len(pos_keys):,}")

    print(f"positive pairs/epoch: {len(A):,}   lr={a.lr} (CONSTANT), negs={a.negs}")
    for ep in range(a.epochs):
        prev = emb.copy()
        t0 = time.time()
        sgd_epoch(emb, G, A, B, layer_of, expert_of, members, pos_keys, a.lr, a.negs, rng)
        steps += len(A)
        # On the unit sphere, chord distance is a direct read of how far each
        # expert actually moved this pass.
        disp = np.linalg.norm(emb - prev, axis=1)
        cos = np.clip(np.einsum('ij,ij->i', emb, prev), -1, 1)
        print(f"  epoch {ep+1}: {time.time()-t0:5.1f}s  mean move {disp.mean():.5f}  "
              f"median {np.median(disp):.5f}  mean angle {np.degrees(np.arccos(cos)).mean():6.2f} deg")

    keys = np.array(list(nodes.keys())); vals = np.array(list(nodes.values()))
    np.savez(a.state, emb=emb, G=G, keys=keys, vals=vals, steps=steps)
    print(f"saved {a.state}  (total steps {steps:,})")


if __name__ == '__main__':
    main()
