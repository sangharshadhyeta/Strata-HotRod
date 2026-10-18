"""Reader for the STRATA_ROUTER_TRACE file: u32 'RTR1', then records  u32 step | u16 layer | u8 n_tok | u8 k | per token: u16[2560] bf16 input, i16[k] ids."""
import struct, sys
import numpy as np

H = 2560


def read_trace(path, max_records=None, step_mod=1):
    """Returns dict of arrays over TOKEN ROWS: step, layer, pos (token index in its window), nt (window size), x (bf16 as uint16 [n, H]), ids int16 [n, k]."""
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    assert bytes(buf[:4]) == b"RTR1", "bad magic"
    off, rows = 4, []
    steps, layers, pos, nts, xs, ids = [], [], [], [], [], []
    n_rec = 0
    size = len(buf)
    while off + 8 <= size:
        step, layer, nt, k = struct.unpack_from("<IHBB", buf, off)
        rec = 8 + nt * (H * 2 + k * 2)
        if off + rec > size:
            break                                   # a record cut by the end of a still-growing file
        if step % step_mod != 0:                    # keep one decode window in step_mod: all 48 layers of a kept step stay together
            off += rec
            continue
        p = off + 8
        for t in range(nt):
            xs.append(np.frombuffer(buf[p:p + H * 2], dtype=np.uint16)); p += H * 2
            ids.append(np.frombuffer(buf[p:p + k * 2], dtype=np.int16)); p += k * 2
            steps.append(step); layers.append(layer); pos.append(t); nts.append(nt)
        off += rec
        n_rec += 1
        if max_records and n_rec >= max_records:
            break
    return dict(step=np.array(steps, np.uint32), layer=np.array(layers, np.uint16), pos=np.array(pos, np.uint8), nt=np.array(nts, np.uint8),
                x=np.stack(xs) if xs else np.zeros((0, H), np.uint16), ids=np.stack(ids) if ids else np.zeros((0, 10), np.int16), records=n_rec)


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


if __name__ == "__main__":
    d = read_trace(sys.argv[1])
    print("records", d["records"], "token rows", len(d["layer"]), "| steps", len(set(d["step"].tolist())))
    ls = np.bincount(d["layer"], minlength=48)
    print("layers seen:", int((ls > 0).sum()), "of 48; rows per layer min/max", int(ls.min()), int(ls.max()))
    print("window sizes:", {int(k): int(v) for k, v in zip(*np.unique(d["nt"], return_counts=True))})
    ids = d["ids"]
    print("ids range", int(ids.min()), int(ids.max()), "| distinct per row (10 expected):", int(np.mean([len(set(r.tolist())) for r in ids[:2000]])))
    x = bf16_to_f32(d["x"][:2000])
    print("x finite:", bool(np.isfinite(x).all()), "| mean |x| %.3f, max |x| %.1f" % (np.abs(x).mean(), np.abs(x).max()))
