"""Tests for trace_reader's token records and token_table_check.py, on small synthetic traces (no GPU, no model).

usage: python test_token_table_check.py
"""
import os
import struct
import tempfile

import numpy as np

import token_table_check as ttc
from trace_reader import H, read_trace

LAYERS = 4


def write_trace(path, windows, with_tokens=True, bad_window=None, rng=None):
    """`windows`: lists of token ids.  Layer 0 routes by the token (its experts are a fixed function of the id), layers 1-3 at random.
    Written in the engine's order: the token record when the window is launched, then the window's 4 layer records."""
    rng = rng or np.random.default_rng(0)
    with open(path, "wb") as f:
        f.write(b"RTR1")
        for step, toks in enumerate(windows, start=1):
            if with_tokens:
                rec_toks = toks + [7] if step == bad_window else toks          # a launch whose row count differs from its window
                f.write(struct.pack("<IHBB", 0, 0xFFFF, len(rec_toks), 0))
                f.write(np.array(rec_toks, np.int32).tobytes())
            for layer in range(LAYERS):
                f.write(struct.pack("<IHBB", step, layer, len(toks), 10))
                for t in toks:
                    f.write(np.zeros(H, np.uint16).tobytes())
                    if layer == 0:
                        ids = [(t * 7 + j * 13) % 512 for j in range(10)]
                    else:
                        ids = rng.choice(512, 10, replace=False).tolist()
                    f.write(np.array(ids, np.int16).tobytes())


def test_reader_pairs_tokens():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        windows = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        write_trace(p, windows, bad_window=2)
        r = read_trace(p, want_x=False)
        assert r["tok_paired"] == 2 and r["tok_mismatch"] == 1, (r["tok_paired"], r["tok_mismatch"])
        assert len(r["x"]) == 0 and len(r["ids"]) == LAYERS * 9
        m = r["step"] == 3
        assert sorted(set(r["tok"][m].tolist())) == [6, 7, 8, 9]
        assert (r["tok"][r["step"] == 2] == -1).all()                   # the mismatched window keeps no tokens
        assert (r["tok"][(r["step"] == 1) & (r["layer"] == 2)] == [1, 2, 3]).all()
        full = read_trace(p)                                            # with the inputs: same rows
        assert full["x"].shape == (LAYERS * 9, H) and (full["tok"] == r["tok"]).all()


def test_old_trace_has_no_tokens():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_trace(p, [[1, 2], [3, 4]], with_tokens=False)
        r = read_trace(p, want_x=False)
        assert r["tok_paired"] == 0 and (r["tok"] == -1).all() and r["records"] == 2 * LAYERS


def test_table_finds_token_routing():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        windows = [rng.integers(0, 40, size=4).tolist() for _ in range(300)]   # 40 distinct tokens: all seen in training
        write_trace(p, windows, rng=rng)
        res = ttc.check(read_trace(p, want_x=False), [0, 3], train_share=0.7)
        assert res[0]["seen"] > 0.99
        assert res[0]["table"][2] > 0.99, res[0]                        # layer 0 follows the token: the table is exact
        assert res[0]["table"][2] > res[0]["freq"][2] + 0.5
        assert res[3]["table"][8] < 0.2, res[3]                         # layer 3 is random: no table can find it


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
