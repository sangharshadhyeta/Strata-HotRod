#!/usr/bin/env python3
"""lowrank_fit.py - fit data-aware low-rank copies of the experts the engine skips most, into an append-only sidecar (lowrank.bin).

A copy replaces a skipped missed expert's contribution with a cheap approximation of the REAL expert: each of its three matrices W (out x in) is factored W ~ B A (B: out x r, A: r x in)
by an SVD whitened with the statistics of the activations the matrix really sees (gate and up: the layer's router input rows; down: the expert's own hidden activations on the same rows),
so the error is spent where the activations are small; the factors are rounded to 4 bits (groups of 32 values, one fp16 scale each).  Offline on Qwen (rank 128, 4-bit, about 24% of an
expert's bytes) the copy's error is 0.55 of skipping the pick in layer 0 and 0.81 in layer 24.

Which experts: --skips is the engine's layer-skips.bin (how often each (layer, expert) was skipped).  Per layer the most-skipped experts without a copy yet are fitted, at most --batch per
run in all, until the copies cover --cover of the layer's skipped picks.  The line printed at the end says how much of each layer is covered; serve/lowrank_maint.py reads it to decide
when the sidecar has settled and stops asking for more.

  tools/lowrank_fit.py --gguf 'DIR/*.gguf' --trace cap-*.bin --skips PACK/layer-skips.bin --out PACK/lowrank.bin --layers 0,1,47 [--rank 128] [--batch 24] [--cover 0.9]

Record format (little endian): file "LRC1" u32 version=1; then records  u16 layer, u16 expert, u16 rank, u16 group; and for gate, up, down in turn, for B then A: the fp16 scales
[rows][ceil(cols/group)] then the packed nibbles [rows][ceil(cols/group)*group/2] (value + 8, low nibble first).  Shapes: gate/up B 640 x r, A r x 2560; down B 2560 x r, A r x 640."""
import argparse, glob, os, struct, sys, time
import numpy as np

MAGIC = b"LRC1"
DIMS = {"gate": (640, 2560), "up": (640, 2560), "down": (2560, 640)}   # (out, in)


def quantize_q4(M, group=32):
    """M (rows x cols) -> (scales fp16 [rows][ngroups], packed uint8 [rows][ngroups*group/2]); symmetric, absmax / 7 per group."""
    rows, cols = M.shape
    ng = (cols + group - 1) // group
    P = np.zeros((rows, ng * group), dtype=np.float32)
    P[:, :cols] = M
    G = P.reshape(rows, ng, group)
    sc = np.abs(G).max(axis=2) / 7.0
    sc16 = sc.astype(np.float16)
    scf = np.maximum(sc16.astype(np.float32), 1e-12)
    q = np.clip(np.rint(G / scf[:, :, None]), -8, 7).astype(np.int8) + 8
    q = q.reshape(rows, ng * group).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return sc16, packed


def dequantize_q4(sc16, packed, cols, group=32):
    rows = packed.shape[0]
    q = np.zeros((rows, packed.shape[1] * 2), dtype=np.int16)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    ng = sc16.shape[1]
    G = (q.reshape(rows, ng, group) - 8).astype(np.float32) * sc16.astype(np.float32)[:, :, None]
    return G.reshape(rows, ng * group)[:, :cols]


def pack_record(layer, expert, rank, factors, group=32):
    """factors: {'gate': (B, A), 'up': (B, A), 'down': (B, A)} float arrays -> record bytes."""
    out = [struct.pack("<HHHH", layer, expert, rank, group)]
    for name in ("gate", "up", "down"):
        for M in factors[name]:
            sc16, packed = quantize_q4(np.asarray(M, dtype=np.float32), group)
            out.append(sc16.tobytes())
            out.append(packed.tobytes())
    return b"".join(out)


def record_size(rank, group=32):
    n = 8
    for name in ("gate", "up", "down"):
        o, i = DIMS[name]
        for rows, cols in ((o, rank), (rank, i)):
            ng = (cols + group - 1) // group
            n += rows * ng * 2 + rows * ng * group // 2
    return n


def read_records(path):
    """[(layer, expert, rank)] of a sidecar, in file order (a damaged tail is ignored)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            return out
        f.read(4)
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            layer, expert, rank, group = struct.unpack("<HHHH", h)
            f.seek(record_size(rank, group) - 8, 1)
            if f.tell() > os.path.getsize(path):    # a record cut short at the end: ignored
                break
            out.append((layer, expert, rank))
    return out


def read_skips(path):
    """{layer: np.array[512]} from the engine's layer-skips.bin ("LSK1", layers, 512, u32 [layers][512])."""
    with open(path, "rb") as f:
        magic, nl, ne = struct.unpack("<III", f.read(12))
        if magic != 0x314B534C:
            raise ValueError("not a layer-skips.bin")
        data = np.frombuffer(f.read(nl * ne * 4), dtype=np.uint32).reshape(nl, ne)
    return {l: data[l].copy() for l in range(nl)}


def pick_experts(counts, have, batch, cover):
    """The experts to fit in one layer: most-skipped first, those without a copy, until `cover` of the layer's skips is reached or `batch` experts are chosen.
    Returns (chosen, covered_share_after)."""
    total = float(counts.sum())
    if total <= 0:
        return [], 1.0
    done = float(sum(counts[e] for e in have))
    chosen = []
    for e in np.argsort(-counts.astype(np.int64), kind="stable"):
        if counts[e] == 0 or len(chosen) >= batch or done / total >= cover:
            break
        if int(e) in have:
            continue
        chosen.append(int(e))
        done += float(counts[e])
    return chosen, done / total


def fit_expert(deq, layer, e, X, rank):
    """The factors of one expert's three matrices against the activations X (n x 2560, float32)."""
    import torch
    silu = lambda z: z / (1 + torch.exp(-z))
    Wg, Wu, Wd = (torch.from_numpy(deq("blk.%d.ffn_%s_exps.weight" % (layer, n), e)) for n in ("gate", "up", "down"))
    Xt = torch.from_numpy(X)
    H = silu(Xt @ Wg.T) * (Xt @ Wu.T)

    def one(M, Z):
        cov = Z.T @ Z / Z.shape[0]
        cov = cov + 1e-3 * float(torch.trace(cov)) / cov.shape[0] * torch.eye(cov.shape[0])
        L = torch.linalg.cholesky(cov)
        U, S, Vh = torch.linalg.svd(M @ L, full_matrices=False)
        return (U[:, :rank] * S[:rank]).numpy(), (Vh[:rank] @ torch.linalg.inv(L)).numpy()
    return {"gate": one(Wg, Xt), "up": one(Wu, Xt), "down": one(Wd, H)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--trace", required=True, nargs="+", help="router-input captures (STRATA_ROUTER_TRACE or the live capture's segments)")
    ap.add_argument("--skips", required=True, help="the engine's layer-skips.bin")
    ap.add_argument("--out", required=True, help="the sidecar (appended to)")
    ap.add_argument("--layers", required=True, help="comma separated layers that get copies")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--batch", type=int, default=24, help="experts fitted per run, over all layers")
    ap.add_argument("--cover", type=float, default=0.9, help="stop adding to a layer once its copies cover this share of its skipped picks")
    ap.add_argument("--rows", type=int, default=700)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor"))
    import trace_reader
    from gguf import GGUFReader, quants
    layers = [int(v) for v in a.layers.split(",")]
    skips = read_skips(a.skips)
    have = {}
    for l, e, r in read_records(a.out):
        have.setdefault(l, set()).add(e)
    plan, cover_after = {}, {}
    budget = a.batch
    for l in layers:
        chosen, cov = pick_experts(skips.get(l, np.zeros(512, dtype=np.uint32)), have.get(l, set()), budget, a.cover)
        plan[l], cover_after[l] = chosen, cov
        budget -= len(chosen)
    todo = [l for l in layers if plan[l]]
    t0 = time.time()
    if todo:
        tens = {}
        for p in sorted(glob.glob(a.gguf)):
            for t in GGUFReader(p, "r").tensors:
                tens[t.name] = t

        def deq(name, e=None):
            t = tens[name]
            d = t.data if e is None else t.data[e]
            if t.tensor_type.name == "F32":
                return np.array(d, dtype=np.float32)
            return quants.dequantize(np.asarray(d), t.tensor_type).astype(np.float32)
        X = {l: [] for l in todo}
        for tp in a.trace:
            for rec in trace_reader.iter_records(tp, want_x=True):
                if rec["kind"] != "decode" or rec["layer"] not in X:
                    continue
                for i in range(rec["n_tok"]):
                    if len(X[rec["layer"]]) < a.rows:
                        X[rec["layer"]].append(rec["x"][i].copy())
            if all(len(v) >= a.rows for v in X.values()):
                break
        new = not os.path.exists(a.out)
        with open(a.out, "ab") as f:
            if new:
                f.write(MAGIC + struct.pack("<I", 1))
            for l in todo:
                x = (np.stack(X[l]).astype(np.uint32) << 16).view(np.float32)
                for e in plan[l]:
                    f.write(pack_record(l, e, a.rank, fit_expert(deq, l, e, x, a.rank)))
                    f.flush()
                print("layer %d: %d copies added (%.0f s)" % (l, len(plan[l]), time.time() - t0), flush=True)
    covered = " ".join("%d:%.3f" % (l, cover_after[l]) for l in layers)
    print("COVERAGE %s | added %d | %.0f s" % (covered, sum(len(v) for v in plan.values()), time.time() - t0))


if __name__ == "__main__":
    main()
