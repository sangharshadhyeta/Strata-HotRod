"""Can a table keyed by the token id predict the first layers' experts?  (22 Oct 2026, offline, no GPU)

Layer 0 has no expert predictor (nothing comes before it) and layer 1's comes one layer late.  Near the bottom of the model a token's
hidden state is still mostly its own embedding, so the experts it routes to may follow from the token id alone: a table
token -> its usual experts at layer l would be known as soon as the window's tokens are (the MTP's drafts included), before the window
starts, at no GPU cost.  This measures that on a router trace that has token records (STRATA_ROUTER_TRACE from an engine with
router_trace_tokens; older traces have none and the script says so).

Split: the trace's windows in order, the first --train share builds the tables, the rest is scored (later text, other prompts).
Per layer, for every scored token row: are its TRUE rank 0-1 experts inside the predicted top K?  (The same metric as README.md's table:
the trained next-layer predictor finds 89.4% at K = 4 on layers 1-47.)
  table   the token's own experts from the training windows (rank 0-1 picks weighted 2, ranks 2-9 weighted 1); a token never seen in
          training falls back to `freq`
  freq    the layer's most used experts in the training windows, the same for every token (what a static hot set gives)
`seen` is the share of scored rows whose token occurs in training at least --min-count times.

usage: python token_table_check.py <trace.bin> [--layers 0-3] [--train 0.7] [--min-count 1]
"""
import argparse
import sys

import numpy as np

from trace_reader import read_trace

E = 512
KS = (2, 4, 8)


def parse_layers(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def check(d, layers, train_share=0.7, min_count=1):
    """Per layer: dict(rows, seen, table={K: recall}, freq={K: recall}).  `d` is read_trace's dict (x not needed)."""
    has = d["tok"] >= 0
    steps = np.unique(d["step"][has])
    if len(steps) < 2:
        return {}
    cut = steps[int(len(steps) * train_share)]
    out = {}
    for l in layers:
        m = has & (d["layer"] == l)
        tok, ids, step = d["tok"][m], d["ids"][m].astype(np.int64), d["step"][m]
        tr, te = step < cut, step >= cut
        if not tr.any() or not te.any():
            continue
        w = np.ones(ids.shape[1])
        w[:2] = 2.0                                                  # rank 0-1 count double
        freq = np.zeros(E)
        np.add.at(freq, ids[tr].ravel(), np.broadcast_to(w, ids[tr].shape).ravel())
        freq_order = np.argsort(-freq, kind="stable")
        table, count = {}, {}
        for t, r in zip(tok[tr], ids[tr]):
            v = table.get(int(t))
            if v is None:
                v = table[int(t)] = np.zeros(E)
            np.add.at(v, r, w)
            count[int(t)] = count.get(int(t), 0) + 1
        hit_t = {k: 0 for k in KS}
        hit_f = {k: 0 for k in KS}
        seen = 0
        n_true = 0
        for t, r in zip(tok[te], ids[te]):
            true = r[:2]
            v = table.get(int(t))
            known = v is not None and count[int(t)] >= min_count
            seen += known
            order = np.argsort(-v, kind="stable") if known else freq_order
            for k in KS:
                hit_t[k] += int(np.isin(true, order[:k]).sum())
                hit_f[k] += int(np.isin(true, freq_order[:k]).sum())
            n_true += len(true)
        n = int(te.sum())
        out[l] = dict(rows=n, seen=seen / n, table={k: hit_t[k] / n_true for k in KS}, freq={k: hit_f[k] / n_true for k in KS})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--min-count", type=int, default=1)
    a = ap.parse_args()
    d = read_trace(a.trace, want_x=False)
    print("windows with tokens: %d paired, %d not (row counts differ)" % (d["tok_paired"], d["tok_mismatch"]))
    if d["tok_paired"] == 0:
        sys.exit("this trace has no token records: record it again with an engine that writes them (router_trace_tokens)")
    res = check(d, parse_layers(a.layers), a.train, a.min_count)
    print("true rank 0-1 experts inside the predicted top K (scored windows: the last %d%%)" % round(100 * (1 - a.train)))
    print("layer   rows   seen | table K=2  K=4  K=8 | freq K=2  K=4  K=8")
    for l, r in res.items():
        print("%5d %6d %5.1f%% |    %5.1f %5.1f %5.1f |   %5.1f %5.1f %5.1f" % (
            l, r["rows"], 100 * r["seen"], *(100 * r["table"][k] for k in KS), *(100 * r["freq"][k] for k in KS)))
    print("reference: the trained next-layer predictor finds 89.4% at K=4 (layers 1-47, README.md); the router one layer early 74.4%")


if __name__ == "__main__":
    main()
