"""Tests for trace_reader on small synthetic traces: RTR2 (request, prefill, decode and token records) and RTR1 (decode only, read as before).
No GPU, no model.

usage: python3 test_trace_reader.py
"""
import os
import struct
import tempfile

import numpy as np

from trace_reader import H, iter_records, read_trace

K = 10


def bf16_row(seed):
    """A bf16 row (uint16) with a known value pattern: element i holds (seed * 1000 + i) as its bf16 bit pattern."""
    return ((np.arange(H, dtype=np.uint32) + seed * 1000) & 0xFFFF).astype(np.uint16)


def write_rtr2(path):
    """Request 1 (a 5-token prompt, prefill of tokens at positions 0, 4 sampled with 1 chunk), its decode window with tokens [11, 12],
    request 2 (a 3-token prompt, sampled at position 6) and its decode window; a token-less decode window of request 2 is not written."""
    with open(path, "wb") as f:
        f.write(b"RTR2")
        # request 1
        f.write(struct.pack("<IHHI", 1, 0xFFFE, 0, 5))
        for layer in range(2):                                  # two MoE layers of the prompt
            pos, tok = [0, 4], [100, 104]
            f.write(struct.pack("<IHHHBB", 1, 0xFFFD, layer, 2, K, 0))
            for j in range(2):
                f.write(struct.pack("<ii", pos[j], tok[j]))
                f.write(bf16_row(10 * layer + j).tobytes())
                f.write(np.array([(layer * 100 + j * 10 + r) for r in range(K)], np.int16).tobytes())
        # decode window of request 1: token record, then layer 0 and 1 records with 2 rows each
        f.write(struct.pack("<IHBB", 0, 0xFFFF, 2, 0))
        f.write(np.array([11, 12], np.int32).tobytes())
        for layer in range(2):
            f.write(struct.pack("<IHBB", 1, layer, 2, K))
            for t in range(2):
                f.write(bf16_row(50 + layer * 2 + t).tobytes())
                f.write(np.array([(t * 7 + r) for r in range(K)], np.int16).tobytes())
        # request 2 (3 prompt tokens, one sampled at absolute position 6), prefill for one layer, then a decode window of one row
        f.write(struct.pack("<IHHI", 2, 0xFFFE, 0, 3))
        f.write(struct.pack("<IHHHBB", 2, 0xFFFD, 1, 1, K, 0))
        f.write(struct.pack("<ii", 6, 777))
        f.write(bf16_row(77).tobytes())
        f.write(np.arange(K, dtype=np.int16).tobytes())
        f.write(struct.pack("<IHBB", 0, 0xFFFF, 1, 0))
        f.write(np.array([55], np.int32).tobytes())
        f.write(struct.pack("<IHBB", 2, 0, 1, K))
        f.write(bf16_row(90).tobytes())
        f.write(np.array([9] * K, np.int16).tobytes())


def write_rtr1(path):
    with open(path, "wb") as f:
        f.write(b"RTR1")
        f.write(struct.pack("<IHBB", 1, 0, 1, K))
        f.write(bf16_row(3).tobytes())
        f.write(np.array([5] * K, np.int16).tobytes())


def test_iter_rtr2_kinds_and_requests():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_rtr2(p)
        recs = list(iter_records(p))
        kinds = [r["kind"] for r in recs]
        assert kinds == ["request", "prefill", "prefill", "tokens", "decode", "decode",
                         "request", "prefill", "tokens", "decode"], kinds
        assert [r["request"] for r in recs] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [r["request"] for r in recs]
        assert recs[0]["n_prompt"] == 5 and recs[6]["n_prompt"] == 3


def test_iter_prefill_contents():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_rtr2(p)
        pf = [r for r in iter_records(p) if r["kind"] == "prefill"]
        a, b, c = pf
        assert (a["layer"], b["layer"], c["layer"]) == (0, 1, 1) and a["n_tok"] == 2 and a["k"] == K
        assert a["pos"].tolist() == [0, 4] and a["tok"].tolist() == [100, 104]
        assert (a["x"][1] == bf16_row(1)).all() and (a["x"][0] == bf16_row(0)).all()
        assert (b["x"][0] == bf16_row(10)).all() and (b["x"][1] == bf16_row(11)).all()
        assert a["ids"][1].tolist() == list(range(10, 20))
        assert b["ids"][0].tolist() == list(range(100, 110))
        assert c["request"] == 2 and c["pos"].tolist() == [6] and c["tok"].tolist() == [777]
        assert (c["x"][0] == bf16_row(77)).all() and c["ids"][0].tolist() == list(range(K))
        no_x = [r for r in iter_records(p, want_x=False) if r["kind"] == "prefill"]
        assert all(r["x"] is None for r in no_x) and no_x[0]["pos"].tolist() == [0, 4]


def test_iter_decode_and_tokens_after_request():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_rtr2(p)
        dec = [r for r in iter_records(p) if r["kind"] == "decode"]
        assert len(dec) == 3 and [r["request"] for r in dec] == [1, 1, 2]
        assert dec[0]["layer"] == 0 and dec[0]["n_tok"] == 2 and (dec[0]["x"][1] == bf16_row(51)).all()
        assert dec[2]["step"] == 2 and dec[2]["ids"][0].tolist() == [9] * K
        toks = [r for r in iter_records(p) if r["kind"] == "tokens"]
        assert toks[0]["tok"].tolist() == [11, 12] and toks[1]["tok"].tolist() == [55]


def test_read_trace_skips_rtr2_only_records():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_rtr2(p)
        r = read_trace(p, want_x=False)
        # 3 decode records (request 1: layers 0 and 1 of 2 rows; request 2: layer 0 of 1 row), prefill and request records skipped
        assert r["records"] == 3 and r["step"].tolist() == [1, 1, 1, 1, 2]
        assert r["layer"].tolist() == [0, 0, 1, 1, 0]
        # both token records pair with their layer-0 records (2 rows and 1 row)
        assert r["tok_paired"] == 2 and r["tok_mismatch"] == 0
        assert r["tok"].tolist() == [11, 12, 11, 12, 55]
        full = read_trace(p)
        assert full["x"].shape == (5, H) and (full["tok"] == r["tok"]).all()
        assert (full["x"][0] == bf16_row(50)).all() and (full["x"][1] == bf16_row(51)).all()


def test_rtr1_still_reads():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_rtr1(p)
        recs = list(iter_records(p))
        assert [r["kind"] for r in recs] == ["decode"] and recs[0]["request"] == 0
        r = read_trace(p)
        assert r["records"] == 1 and r["layer"].tolist() == [0] and (r["x"][0] == bf16_row(3)).all()
        assert r["ids"][0].tolist() == [5] * K


def test_truncated_prefill_record_is_dropped():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_rtr2(p)
        size = os.path.getsize(p)
        with open(p, "r+b") as f:
            f.truncate(size - 100)                  # cut the last decode record: it is dropped, the records before it are read
        kinds = [r["kind"] for r in iter_records(p)]
        assert kinds == ["request", "prefill", "prefill", "tokens", "decode", "decode",
                         "request", "prefill", "tokens"], kinds


def test_bad_magic_is_refused():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        with open(p, "wb") as f:
            f.write(b"XXXX" + b"\0" * 16)
        try:
            list(iter_records(p))
        except ValueError:
            return
        raise AssertionError("bad magic was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
