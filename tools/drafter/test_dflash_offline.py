"""Tests for dflash_offline.py's trace parsing, today's numbers and the replay loop, on synthetic STRATA_DRAFT_TRACE files
(no GPU, no model, no download).

usage: python test_dflash_offline.py
"""
import os
import struct
import tempfile

import numpy as np

import dflash_offline as dfo

L, N = 5, 8          # a tiny "hidden size": the format does not care


def feat(pos):
    """A position's features, recognisable in the rebuilt request: every value is the position (as bf16 bits)."""
    return np.full(L * N, pos & 0x7FFF, np.uint16)


def emulate(text, prompt, accepts, T=4, rng=None):
    """The windows an engine writes for one request: the prompt in windows of T rows, then generating windows whose drafts the
    model accepts `accepts[i]` of (the rest of the window wrong).  -> list of (pos0, tokens, picks, feats)."""
    rng = rng or np.random.default_rng(0)
    wrong = lambda: int(rng.integers(50_000, 60_000))
    wins = []
    for p0 in range(0, prompt, T):
        rows = list(range(p0, min(p0 + T, prompt)))
        toks = [text[p] for p in rows]
        picks = [text[p + 1] if p + 1 == prompt else wrong() for p in rows]    # only the last prompt row's pick is kept
        wins.append((p0, toks, picks, [feat(p) for p in rows]))
    s, i = prompt, 0
    while s + 1 < len(text) and i < len(accepts):
        a = min(accepts[i], T - 1, len(text) - s - 2)
        toks = [text[s]] + [text[s + 1 + j] for j in range(a)] + [wrong() for _ in range(T - 1 - a)]
        picks = [text[s + 1 + j] for j in range(a + 1)] + [wrong() for _ in range(T - 1 - a)]
        wins.append((s, toks, picks, [feat(s + j) for j in range(T)]))   # rejected rows hold the features of wrong tokens too
        s += a + 1
        i += 1
    return wins


def write(path, requests):
    with open(path, "wb") as f:
        f.write(b"DFT1" + struct.pack("<III", L, N, 4) + struct.pack("<%di" % L, 5, 15, 25, 35, 45))
        for wins in requests:
            for pos0, toks, picks, feats in wins:
                f.write(struct.pack("<qi", pos0, len(toks)))
                f.write(np.array(toks, np.int32).tobytes() + np.array(picks, np.int32).tobytes())
                f.write(np.stack(feats).astype(np.uint16).tobytes())


def test_parse_rebuild_and_today():
    text = list(range(100, 140))
    accepts = [3, 0, 2, 3, 1, 3, 3, 0, 2]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write(p, [emulate(text, 10, accepts), emulate(list(range(200, 220)), 6, [1, 1, 1])])
        layers, n_embd, recs = dfo.read_trace(p)
        assert layers == [5, 15, 25, 35, 45] and n_embd == N
        seqs = dfo.sequences(recs)
        assert len(seqs) == 2
        s = seqs[0]
        assert s["gen_start"] == 10, s["gen_start"]
        n = len(s["tokens"])
        assert list(s["tokens"]) == text[:n], "the committed text is rebuilt (rejected drafts rewritten)"
        assert all(int(s["feats"][q][0]) == q for q in range(n)), "every position keeps its committed row's features"
        assert len(s["gen_windows"]) == len(accepts)
        t = dfo.today_stats(seqs[:1])
        assert abs(t["tokens_per_window"] - (np.mean(accepts) + 1)) < 1e-9, t
        assert seqs[1]["gen_start"] == 6 and list(seqs[1]["tokens"]) == list(range(200, 200 + len(seqs[1]["tokens"])))


def test_replay_counts_what_greedy_accepts():
    text = list(range(100, 160))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write(p, [emulate(text, 12, [3] * 30)])
        seq = dfo.sequences(dfo.read_trace(p)[2])[0]
        toks = seq["tokens"]
        n = len(toks)
        seen = []

        def oracle(ctx_lo, ctx_hi, start, anchor, block):          # always right: every round yields the whole block
            assert anchor == toks[start] and ctx_hi == start and ctx_lo <= ctx_hi
            seen.append((ctx_lo, ctx_hi))
            return [int(toks[start + 1 + j]) if start + 1 + j < n else -1 for j in range(block - 1)]

        rounds = dfo.replay(seq, oracle, 4)
        assert sum(rounds) == n - 1 - seq["gen_start"], (sum(rounds), n, seq["gen_start"])
        assert all(r == 4 for r in rounds[:-1])
        assert seen[0] == (0, 12), "the first round gets the whole prompt's features"
        assert all(seen[i][0] == seen[i - 1][1] for i in range(1, len(seen))), "each round adds only the positions the last one committed"

        def first_right(ctx_lo, ctx_hi, start, anchor, block):     # only the first draft right: 2 tokens a round
            return [int(toks[start + 1])] + [-1] * (block - 2)

        rounds = dfo.replay(seq, first_right, 8)
        assert all(r == 2 for r in rounds[:-1]) and sum(rounds) == n - 1 - seq["gen_start"]


def test_trace_without_prompt_features():
    text = list(range(300, 330))
    wins = [w for w in emulate(text, 8, [2] * 10) if w[0] >= 8]   # a --prefill run: the prompt's windows are not in the trace
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write(p, [wins])
        seq = dfo.sequences(dfo.read_trace(p)[2])[0]
        assert seq["first"] == 8 and seq["gen_start"] == 0
        rounds = dfo.replay(seq, lambda lo, hi, st, an, b: [-1] * (b - 1), 4)
        assert all(r == 1 for r in rounds)


def test_cut_file_and_bad_magic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write(p, [emulate(list(range(100, 130)), 8, [1] * 5)])
        full = len(dfo.read_trace(p)[2])
        with open(p, "r+b") as f:
            f.truncate(os.path.getsize(p) - 10)
        assert len(dfo.read_trace(p)[2]) == full - 1
        with open(p, "wb") as f:
            f.write(b"RTR1" + bytes(40))
        try:
            dfo.read_trace(p)
            raise AssertionError("a router trace was taken for a draft trace")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
