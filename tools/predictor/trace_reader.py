"""Reader for the STRATA_ROUTER_TRACE file.  Two formats, one reader:

RTR1 (decode only): u32 'RTR1', then decode records  u32 step | u16 layer | u8 n_tok | u8 k | per token: u16[2560] bf16 input, i16[k] ids.
Token records (both formats): u32 0 | u16 0xFFFF | u8 n_tok | u8 0 | i32[n_tok] token ids, written as each decode window is launched; they are
paired with the windows in order (a pair whose row counts differ is left without tokens: `tok` -1).

RTR2 (engine with STRATA_ROUTER_TRACE_PREFILL=1): u32 'RTR2', then the same records and two more kinds, all with a 12-byte header:
  request  u32 request | u16 0xFFFE | u16 0 | u32 n_prompt          (written at the start of each prompt; decode windows after it belong to it)
  prefill  u32 request | u16 0xFFFD | u16 layer | u16 n_tok | u8 k | u8 0, then per sampled prompt token:
           i32 pos (absolute position in the conversation) | i32 token id | u16[2560] bf16 router input | i16[k] ids (the router's own top-k)
"""
import struct, sys
import numpy as np

H = 2560
MARK_TOKENS, MARK_REQUEST, MARK_PREFILL = 0xFFFF, 0xFFFE, 0xFFFD


def _is_rtr2(buf):
    magic = bytes(buf[:4])
    if magic not in (b"RTR1", b"RTR2"):
        raise ValueError("bad magic %r: not a router trace" % magic)
    return magic == b"RTR2"


def iter_records(path, want_x=True):
    """Yields one dict per record, in file order.  kind is 'decode', 'tokens', 'request' or 'prefill'; `request` is the number of the request the
    record belongs to (0 before the first request record, and always 0 in an RTR1 file).
      decode:  step, layer, n_tok, k, ids int16 [n_tok, k], x uint16 [n_tok, H] (None when want_x is False)
      tokens:  tok int32 [n_tok]
      request: n_prompt
      prefill: layer, n_tok, k, pos int32 [n_tok], tok int32 [n_tok], ids int16 [n_tok, k], x uint16 [n_tok, H] (None when want_x is False)
    A record cut off by the end of a still-growing file ends the iteration."""
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    rtr2 = _is_rtr2(buf)
    size, off, request = len(buf), 4, 0
    while off + 8 <= size:
        step, layer, nt, k = struct.unpack_from("<IHBB", buf, off)
        if layer == MARK_TOKENS:
            if off + 8 + nt * 4 > size:
                return
            tok = np.frombuffer(buf[off + 8:off + 8 + nt * 4].tobytes(), dtype=np.int32).copy()
            yield dict(kind="tokens", request=request, tok=tok)
            off += 8 + nt * 4
        elif layer == MARK_REQUEST and rtr2:
            if off + 12 > size:
                return
            request = step
            yield dict(kind="request", request=request, n_prompt=struct.unpack_from("<I", buf, off + 8)[0])
            off += 12
        elif layer == MARK_PREFILL and rtr2:
            req, _, lay, n, kk, _ = struct.unpack_from("<IHHHBB", buf, off)
            dt = np.dtype([("pos", "<i4"), ("tok", "<i4"), ("x", "<u2", (H,)), ("ids", "<i2", (kk,))])
            if off + 12 + n * dt.itemsize > size:
                return
            rows = np.frombuffer(buf[off + 12:off + 12 + n * dt.itemsize].tobytes(), dtype=dt)
            yield dict(kind="prefill", request=req, layer=lay, n_tok=n, k=kk, pos=rows["pos"].copy(), tok=rows["tok"].copy(),
                       ids=rows["ids"].copy(), x=rows["x"].copy() if want_x else None)
            off += 12 + n * dt.itemsize
        else:
            dt = np.dtype([("x", "<u2", (H,)), ("ids", "<i2", (k,))])
            if off + 8 + nt * dt.itemsize > size:
                return                                  # a record cut by the end of a still-growing file
            rows = np.frombuffer(buf[off + 8:off + 8 + nt * dt.itemsize].tobytes(), dtype=dt)
            yield dict(kind="decode", request=request, step=step, layer=layer, n_tok=nt, k=k, ids=rows["ids"].copy(),
                       x=rows["x"].copy() if want_x else None)
            off += 8 + nt * dt.itemsize


def read_trace(path, max_records=None, step_mod=1, want_x=True):
    """Returns dict of arrays over TOKEN ROWS of the decode records: step, layer, pos (token index in its window), nt (window size), x (bf16 as uint16
    [n, H]), ids int16 [n, k], tok int32 (the row's token id, -1 when the trace has no token records or the pairing failed); also tok_paired /
    tok_mismatch (windows).  Prefill and request records are skipped (see iter_records).
    want_x=False leaves x empty (the inputs are most of the file; tools that need only the routing skip them)."""
    steps, layers, pos, nts, xs, ids = [], [], [], [], [], []
    pending, step_tok = [], {}                      # token records not yet paired; step -> that window's token ids
    paired = mismatch = 0
    n_rec = 0
    for r in iter_records(path, want_x=want_x):
        if r["kind"] == "tokens":
            pending.append(r["tok"])
            continue
        if r["kind"] != "decode":
            continue
        step, layer, nt = r["step"], r["layer"], r["n_tok"]
        if layer == 0 and pending:                  # a new window: the oldest launch not yet paired is this one
            toks = pending.pop(0)
            if len(toks) == nt:
                step_tok[step] = toks
                paired += 1
            else:
                mismatch += 1
        if step % step_mod != 0:                    # keep one decode window in step_mod: all 48 layers of a kept step stay together
            continue
        ids.append(r["ids"])
        if want_x:
            xs.append(r["x"])
        steps.extend([step] * nt); layers.extend([layer] * nt); pos.extend(range(nt)); nts.extend([nt] * nt)
        n_rec += 1
        if max_records and n_rec >= max_records:
            break
    tok = np.array([int(step_tok[s][p]) if s in step_tok else -1 for s, p in zip(steps, pos)], np.int32)
    return dict(step=np.array(steps, np.uint32), layer=np.array(layers, np.uint16), pos=np.array(pos, np.uint8), nt=np.array(nts, np.uint8),
                x=np.concatenate(xs) if xs else np.zeros((0, H), np.uint16), ids=np.concatenate(ids) if ids else np.zeros((0, 10), np.int16),
                records=n_rec, tok=tok, tok_paired=paired, tok_mismatch=mismatch)


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


if __name__ == "__main__":
    d = read_trace(sys.argv[1])
    print("records", d["records"], "token rows", len(d["layer"]), "| steps", len(set(d["step"].tolist())))
    ls = np.bincount(d["layer"], minlength=48)
    print("layers seen:", int((ls > 0).sum()), "of 48; rows per layer min/max", int(ls.min()), int(ls.max()))
    print("token records: %d windows paired, %d with a different row count" % (d["tok_paired"], d["tok_mismatch"]))
    print("window sizes:", {int(k): int(v) for k, v in zip(*np.unique(d["nt"], return_counts=True))})
    ids = d["ids"]
    print("ids range", int(ids.min()), int(ids.max()), "| distinct per row (10 expected):", int(np.mean([len(set(r.tolist())) for r in ids[:2000]])))
    x = bf16_to_f32(d["x"][:2000])
    print("x finite:", bool(np.isfinite(x).all()), "| mean |x| %.3f, max |x| %.1f" % (np.abs(x).mean(), np.abs(x).max()))
    kinds = {}
    prefill_rows = {}
    for r in iter_records(sys.argv[1], want_x=False):
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        if r["kind"] == "prefill":
            prefill_rows[r["request"]] = prefill_rows.get(r["request"], 0) + r["n_tok"]
    print("records by kind:", kinds)
    if prefill_rows:
        print("prefill token rows (layer records) per request:", prefill_rows)
