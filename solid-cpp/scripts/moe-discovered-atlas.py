#!/usr/bin/env python3
"""Track 1.6 - build an atlas whose axes are DISCOVERED, not assigned.

The existing atlas positions each expert against 9 categories hardcoded in
k_probes[] ("code", "math", "law", ...). Those are human guesses about what a
model ought to specialise in, measured offline with 27 probes. This builds the
same kind of map from what the router actually did, with no category list at
all: factor the co-activation graph and use its components as the axes.

Why co-activation and not req_dir: updating an atlas from req_dir is
self-training, because req_dir is itself inferred from the atlas - a slightly
wrong estimate reinforces itself (docs/plan.md, Track 1.6). "These two experts
fired in the same routing decision" is a direct observation with no atlas in
the loop, so there is no circularity to break.

The graph is GLOBAL over all (layer, expert) nodes rather than one embedding
per layer. Per-layer factorings each come out in their own arbitrary rotation,
so coordinates would not be comparable between layers - and req_dir averages
positions across layers, so it needs one shared space. Cross-layer edges
(layer L's top pick -> layer L+1's top pick, step 4b) are what tie the layers
into that shared space.

Output matches the existing atlas schema (x, y per cell) so it is a drop-in
for everything that already reads one, plus a `dims` vector holding more
components - the discovered analogue of `cats{}`.
"""
import json, sys, argparse
from collections import defaultdict
import numpy as np

K = 8


def load_traces(paths, keep=K):
    """Returns per-file lists of (layer, [expert ids]) in decode order.

    Was hardcoded to the 4-topic traces/deep-*.txt set (28,800 decisions) -
    the trace files actually used for the 31.6x-data rebuild that got the
    spectral method to its real measured ceiling (moe-link-prediction.py,
    AUC 0.5706-0.5761 on 909,480 pooled decisions) are traces/big-*.txt (12
    topics). The on-disk expert-atlas-discovered.json this script produces
    was never regenerated from that larger set - --traces makes that
    explicit instead of silently defaulting to the smaller one.
    """
    out = {}
    for p in paths:
        seq = []
        for line in open(p):
            f = line.split()
            seq.append((int(f[0]), [int(x.split(':')[0]) for x in f[2:2 + keep]]))
        out[p] = seq
    return out


def build_graph(streams, n_expert=256):
    """Weighted undirected graph over (layer, expert).

    within-layer : experts selected in the SAME routing decision (step 4a)
    cross-layer  : consecutive layers' top picks for the same token (step 4b)
    """
    edges = defaultdict(float)
    nodes = {}

    def nid(l, e):
        k = (l, e)
        if k not in nodes:
            nodes[k] = len(nodes)
        return nodes[k]

    for seq in streams.values():
        prev = None
        for il, picks in seq:
            for i in range(len(picks)):
                a = nid(il, picks[i])
                for j in range(i + 1, len(picks)):
                    b = nid(il, picks[j])
                    if a != b:
                        edges[(min(a, b), max(a, b))] += 1.0
            if prev is not None and prev[0] != il:
                a, b = nid(prev[0], prev[1]), nid(il, picks[0])
                if a != b:
                    edges[(min(a, b), max(a, b))] += 1.0
            prev = (il, picks[0])
    return nodes, edges


def embed(nodes, edges, dims=8, iters=12, seed=0):
    """Top eigenvectors of the normalised adjacency, by randomised subspace
    iteration - keeps everything to sparse matvecs so a 7680-node graph never
    needs a dense 7680x7680 factorisation."""
    n = len(nodes)
    r = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    c = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    w = np.fromiter(edges.values(), dtype=np.float64, count=len(edges))

    deg = np.zeros(n)
    np.add.at(deg, r, w)
    np.add.at(deg, c, w)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    wn = w * dinv[r] * dinv[c]          # D^-1/2 A D^-1/2

    def matvec(X):
        Y = np.zeros_like(X)
        np.add.at(Y, r, wn[:, None] * X[c])
        np.add.at(Y, c, wn[:, None] * X[r])
        return Y

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, dims + 4)))
    for _ in range(iters):
        Q, _ = np.linalg.qr(matvec(Q))
    B = Q.T @ matvec(Q)
    vals, vecs = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(-np.abs(vals))
    return (Q @ vecs[:, order])[:, :dims], vals[order][:dims], deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', nargs='+', required=True)
    ap.add_argument('--out', default='/mnt/nvme/models/ornith/expert-atlas-discovered.json')
    ap.add_argument('--dims', type=int, default=8)
    a = ap.parse_args()

    streams = load_traces(a.traces)
    nodes, edges = build_graph(streams)
    print(f"graph: {len(nodes)} nodes, {len(edges)} edges", file=sys.stderr)
    emb, vals, deg = embed(nodes, edges, dims=a.dims)
    print(f"top eigenvalues: {np.round(vals, 4)}", file=sys.stderr)

    # Component 0 of a normalised adjacency is the degree vector - it encodes
    # "how often was this expert used at all", which the heat counters already
    # track. The informative axes start at 1.
    xy = emb[:, 1:3]

    # Raw eigenvector magnitudes are heavy-tailed (median magnitude ~0.1x the
    # max here), so dividing by the single largest point crushes most of the
    # disc into an unreadable dot at the centre - measured directly: half of
    # all cells landed within 10% of the max radius.
    #
    # First attempt kept each point's angle and rank-transformed only the
    # radius (magnitude), which fixed the crush but created a "wheel" of
    # spokes: many experts share similar (dim1, dim2) angles (real graph
    # clusters), so stretching those clusters out by radius alone turns each
    # one into a radial line instead of a filled region. Rank-normalising x
    # and y INDEPENDENTLY instead (quantile normalisation to a uniform
    # marginal on each axis) fills the square/disc by area without imposing
    # that artificial polar structure - a point's relative order along each
    # axis is preserved, its exact radius/angle from the raw eigenvectors is
    # not, which is the right trade for a readable scatter.
    def rank_normalize(v):
        order = np.argsort(v)
        rank = np.empty_like(order, dtype=np.float64)
        rank[order] = np.arange(len(v))
        return 2.0 * (rank + 0.5) / len(v) - 1.0  # -> uniform on (-1, 1)

    sq = np.stack([rank_normalize(xy[:, 0]), rank_normalize(xy[:, 1])], axis=1)

    # rank_normalize fills a SQUARE [-1,1]x[-1,1], but the UI draws a CIRCLE -
    # corner points (magnitude up to sqrt(2)) landed outside it. Shirley-Chiu
    # concentric map: a standard, closed-form square->disc transform that
    # takes a uniform-in-square point to a uniform-in-disc point, so every
    # point ends up inside the circle without re-introducing the crushed- or
    # wheel-shaped artifacts of the earlier attempts.
    sx, sy = sq[:, 0], sq[:, 1]
    r = np.where(np.abs(sx) > np.abs(sy), sx, sy)
    theta = np.where(
        np.abs(sx) > np.abs(sy),
        (np.pi / 4) * np.divide(sy, sx, out=np.zeros_like(sy), where=sx != 0),
        (np.pi / 2) - (np.pi / 4) * np.divide(sx, sy, out=np.zeros_like(sx), where=sy != 0),
    )
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    xy[(sx == 0) & (sy == 0)] = 0.0

    cells = []
    for (l, e), i in nodes.items():
        cells.append({
            "layer": l, "expert": e,
            "n": int(deg[i]),
            "spec": round(float(np.abs(xy[i]).max()), 4),
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "dims": [round(float(v), 4) for v in emb[i, 1:a.dims]],
        })
    cells.sort(key=lambda c: (c['layer'], c['expert']))
    json.dump({"categories": [f"d{i}" for i in range(1, a.dims)],
               "n_layer": max(c['layer'] for c in cells) + 1,
               "n_probes": 0,
               "discovered": True,
               "cells": cells}, open(a.out, 'w'))
    print(f"wrote {len(cells)} cells to {a.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
