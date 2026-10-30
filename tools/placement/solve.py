"""Offline expert placement for Strata-HotRod (the Taalas lesson: decide once, from real routing, where each expert lives).

From a STRATA_ROUTER_TRACE recorded with the stand-ins OFF (the model's own top-k, rank order), each (layer, expert) gets a score: its picks at
the exact ranks (below the stand-in floor: a miss there is a read) plus beta x its picks at the ranks the stand-ins serve (a miss there mostly
takes a stand-in already on the card).  The best scores go to the card, the next to the RAM tier, the rest stay on the SSD.  The engine's
expert profile (STRP v1, read by --expert-profile) is one global ranked list, so the layers get the slots their routing needs.

  solve.py --train rt_train.bin --eval rt_eval.bin --card 2586 --ram 9202 [--profile old.bin] [--out new.bin]

prints, for the eval trace, the exact-rank picks the card serves / the RAM tier serves / the SSD reads, for: the old profile, plain counts
(every pick alike) and the rank-weighted score; --out writes the rank-weighted ranking as a profile."""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "predictor"))
from trace_reader import iter_records  # noqa: E402


def counts(path, n_layers, n_expert, floor):
    """[layer, expert] picks below the floor (exact) and at or above it (stand-in ranks), over the trace's decode records."""
    ex = np.zeros((n_layers, n_expert), np.int64)
    tl = np.zeros((n_layers, n_expert), np.int64)
    for r in iter_records(path, want_x=False):
        if r["kind"] != "decode":
            continue
        ids = np.asarray(r["ids"], np.int64)
        lay = int(r["layer"])
        if lay >= n_layers:
            continue
        for rank in range(ids.shape[1]):
            col = ids[:, rank]
            col = col[(col >= 0) & (col < n_expert)]
            np.add.at(ex if rank < floor else tl, (lay, col), 1)
    return ex, tl


def read_profile(path, n_layers, n_expert):
    """The ranked (layer, expert) list of an STRP v1 profile."""
    b = open(path, "rb").read()
    if b[:4] != b"STRP":
        raise ValueError(path + " is not an STRP profile")
    version, nl, ne, want, n = struct.unpack_from("<5I", b, 4)
    if version != 1 or nl != n_layers or ne != n_expert:
        raise ValueError(f"{path}: version {version}, {nl} x {ne} (want 1, {n_layers} x {n_expert})")
    pairs = np.frombuffer(b, dtype="<u2", count=2 * n, offset=24).reshape(n, 2)   # (u16 layer, u16 expert) pairs
    return [(int(l), int(e)) for l, e in pairs]


def write_profile(path, n_layers, n_expert, ranked, want):
    with open(path, "wb") as f:
        f.write(b"STRP" + struct.pack("<5I", 1, n_layers, n_expert, want, len(ranked)))
        f.write(np.asarray(ranked, dtype="<u2").tobytes())   # (u16 layer, u16 expert) pairs, as read_expert_profile reads them


def ranking(score):
    """Every (layer, expert), best score first (ties: lower layer, then lower expert)."""
    nl, ne = score.shape
    flat = score.reshape(-1)
    order = np.lexsort((np.arange(flat.size), -flat))
    return [(int(i // ne), int(i % ne)) for i in order]


def evaluate(ranked, ex_eval, card, ram):
    """The eval trace's exact-rank picks served by the card, by the RAM tier, and read from the SSD, for a static placement."""
    tier = np.full(ex_eval.shape, 2, np.int8)
    for i, (l, e) in enumerate(ranked):
        tier[l, e] = 0 if i < card else 1 if i < card + ram else 2
    tot = ex_eval.sum()
    return [float(ex_eval[tier == t].sum()) / max(1, tot) for t in (0, 1, 2)], int(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--card", type=int, required=True, help="the card's expert slots")
    ap.add_argument("--ram", type=int, required=True, help="the experts the RAM tier holds")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--floor", type=int, default=3, help="ranks below it are exact (the stand-in floor)")
    ap.add_argument("--beta", type=float, default=0.25, help="the weight of a pick at a stand-in rank")
    ap.add_argument("--profile", help="the current profile, for comparison")
    ap.add_argument("--out", help="write the rank-weighted ranking here (STRP v1)")
    a = ap.parse_args()
    ex_tr, tl_tr = counts(a.train, a.layers, a.experts, a.floor)
    ex_ev, _ = counts(a.eval, a.layers, a.experts, a.floor)
    arms = {}
    if a.profile:
        arms["current profile"] = read_profile(a.profile, a.layers, a.experts)
    arms["plain counts"] = ranking((ex_tr + tl_tr).astype(np.float64))
    arms[f"rank-weighted (beta {a.beta})"] = ranking(ex_tr + a.beta * tl_tr)
    print(f"train: {int(ex_tr.sum() + tl_tr.sum())} picks; eval: {int(ex_ev.sum())} exact-rank picks; card {a.card}, RAM {a.ram} experts")
    print(f"{'placement':28} {'card':>7} {'RAM':>7} {'SSD':>7}")
    for name, ranked in arms.items():
        (c, r, s), _ = evaluate(ranked, ex_ev, a.card, a.ram)
        print(f"{name:28} {100 * c:6.1f}% {100 * r:6.1f}% {100 * s:6.1f}%")
    if a.out:
        write_profile(a.out, a.layers, a.experts, arms[f"rank-weighted (beta {a.beta})"], a.layers * a.experts)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
