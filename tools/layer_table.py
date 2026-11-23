#!/usr/bin/env python3
"""layer_table.py - make and refresh the per-layer sensitivity table (<pack>/layer-sens.txt) that STRATA_TAIL_WEIGHT scales its threshold with.

For each MoE layer it measures, on real router inputs (a STRATA_ROUTER_TRACE capture of your own traffic) and the model's real expert outputs (computed here from the GGUF):
how much the block output changes when 10% of the picks are dropped, smallest router weight first (ranks below --from-rank are never dropped).  A layer whose output moves
little is tolerant (the engine skips more of its small picks); one that moves a lot is sensitive (fewer).  The engine reads the file at the next request, so the table can be
refreshed while it serves; with no file every layer keeps the flat threshold, which is the shipped default.

The table improves with use: each run blends its measurement into the file already there (exponential average, --alpha), and --part k/n measures only every n-th layer, so a
short job can run from time to time (it is CPU work: run it with `nice`, and not while a speed test runs).

  tools/layer_table.py --trace TRACE.bin --gguf 'DIR/*.gguf' --out PACK/layer-sens.txt [--rows 600] [--alpha 0.3] [--part 0/4] [--from-rank 3]

How to get a trace: start the server with STRATA_ROUTER_TRACE=<file> for a few hundred decode tokens (it records the router input and the model's top-10 per layer and window;
it costs speed while it is on, so it is a measurement run, not a setting to leave on).  Format of the output: one number per layer, '#' lines are comments.
"""
import argparse, glob, os, sys, time


def parse_table(text):
    """The numbers of a table file, in order ('#' lines ignored)."""
    out = []
    for ln in text.splitlines():
        if ln.startswith("#"):
            continue
        out.extend(float(x) for x in ln.split())
    return out


def parse_measured(text):
    """The layers that really were measured (the others hold the mean as a placeholder), from the '# measured layers:' line."""
    for ln in text.splitlines():
        if ln.startswith("# measured layers:"):
            return set(int(x) for x in ln.split(":", 1)[1].split())
    return None   # a hand-made table: every number counts


def merge_table(old, new, alpha, measured_old=None):
    """Blend a fresh measurement into the table already there.  new[l] is None for a layer this run did not measure.
    No old value (or a table of another size, or a placeholder): the measurement itself.  alpha is the weight of the new measurement."""
    out = []
    for l, v in enumerate(new):
        o = old[l] if old is not None and l < len(old) and len(old) == len(new) and (measured_old is None or l in measured_old) else None
        if v is None:
            out.append(o)                      # not measured now: keep what was there (None if nothing)
        elif o is None:
            out.append(v)
        else:
            out.append((1.0 - alpha) * o + alpha * v)
    return out


def format_table(vals, runs, note):
    head = ["# per-layer sensitivity for STRATA_TAIL_WEIGHT (tools/layer_table.py)",
            "# block-output error with 10% of the picks dropped by smallest weight; larger = the layer suffers more, so the engine skips less of it",
            "# runs blended into this table: %d   %s" % (runs, note),
            "# measured layers: " + " ".join(str(l) for l, v in enumerate(vals) if v is not None)]
    present = [v for v in vals if v is not None]
    fill = sum(present) / len(present) if present else 1.0
    return "\n".join(head + ["%.5f" % (v if v is not None else fill) for v in vals]) + "\n"


def parse_runs(text):
    for ln in text.splitlines():
        if ln.startswith("# runs blended into this table:"):
            try:
                return int(ln.split(":")[1].split()[0])
            except Exception:
                return 0
    return 0


def measure(args, layers):
    """{layer: relative block-output error at 10% dropped by weight}.  Needs numpy, torch, gguf and the repo's trace reader."""
    import numpy as np, torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor"))
    import trace_reader
    from gguf import GGUFReader, quants
    torch.set_num_threads(args.threads)
    files = sorted(glob.glob(args.gguf))
    if not files:
        sys.exit("no GGUF files match " + args.gguf)
    tens = {}
    for p in files:
        for t in GGUFReader(p, "r").tensors:
            tens[t.name] = t

    def deq(name, e=None):
        t = tens[name]
        d = t.data if e is None else t.data[e]
        if t.tensor_type.name == "F32":
            return np.array(d, dtype=np.float32)
        return quants.dequantize(np.asarray(d), t.tensor_type).astype(np.float32)

    X = {l: [] for l in layers}
    IDS = {l: [] for l in layers}
    for rec in trace_reader.iter_records(args.trace, want_x=True):
        if rec["kind"] != "decode" or rec["layer"] not in X:
            continue
        l = rec["layer"]
        for i in range(rec["n_tok"]):
            if len(X[l]) < args.rows:
                X[l].append(rec["x"][i].copy())
                IDS[l].append(rec["ids"][i].astype(np.int64))
        if all(len(X[k]) >= args.rows for k in X):
            break
    silu = lambda z: z / (1 + torch.exp(-z))
    res = {}
    for l in layers:
        if len(X[l]) < 50:
            print("layer %d: only %d rows in the trace, skipped" % (l, len(X[l])), flush=True)
            continue
        W = deq("blk.%d.ffn_gate_inp.weight" % l).reshape(-1, 2560)
        x = (np.stack(X[l]).astype(np.uint32) << 16).view(np.float32)
        ids = np.stack(IDS[l])
        N = len(x)
        lg = x @ W.T
        sel = np.take_along_axis(lg, ids, axis=1)
        w = np.exp(sel - sel.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        Xt = torch.from_numpy(x)
        Y = torch.zeros(N, ids.shape[1], x.shape[1])
        for e in np.unique(ids):
            tr, rk = np.where(ids == e)
            Wg, Wu, Wd = (torch.from_numpy(deq("blk.%d.ffn_%s_exps.weight" % (l, n), e)) for n in ("gate", "up", "down"))
            xe = Xt[tr]
            Y[tr, rk] = (silu(xe @ Wg.T) * (xe @ Wu.T)) @ Wd.T
        Sg, Su, Sd = (torch.from_numpy(deq("blk.%d.ffn_%s_shexp.weight" % (l, n))) for n in ("gate", "up", "down"))
        sgate = torch.from_numpy(deq("blk.%d.ffn_gate_inp_shexp.weight" % l).reshape(-1))
        shared = torch.sigmoid(Xt @ sgate)[:, None] * ((silu(Xt @ Sg.T) * (Xt @ Su.T)) @ Sd.T)
        wt = torch.from_numpy(w.astype(np.float32))
        C = Y * wt[:, :, None]
        nf = (C.sum(1) + shared).norm(dim=1)
        key = np.where(np.arange(ids.shape[1])[None, :] >= args.from_rank, w, np.inf).ravel()
        n_drop = int(round(0.10 * N * ids.shape[1]))
        d = np.zeros(N * ids.shape[1], bool)
        d[np.argsort(key, kind="stable")[:n_drop]] = True
        d = torch.from_numpy(d.reshape(N, ids.shape[1]))
        res[l] = float(((C * d[:, :, None]).sum(1).norm(dim=1) / nf).mean())
        print("layer %2d: error at 10%% dropped by weight %.4f (%d rows)" % (l, res[l], N), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="a STRATA_ROUTER_TRACE capture")
    ap.add_argument("--gguf", required=True, help="glob of the model's GGUF files")
    ap.add_argument("--out", required=True, help="the table (<pack>/layer-sens.txt); blended into when it exists")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--rows", type=int, default=600, help="decode rows per layer from the trace")
    ap.add_argument("--alpha", type=float, default=0.3, help="weight of this run's measurement when blending into an existing table")
    ap.add_argument("--part", default="0/1", help="k/n: measure only layers l with l %% n == k (a short job per run)")
    ap.add_argument("--from-rank", type=int, default=3, help="ranks below this are never dropped (the stand-ins' floor)")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    k, n = (int(v) for v in a.part.split("/"))
    layers = [l for l in range(a.layers) if l % n == k]
    t0 = time.time()
    res = measure(a, layers)
    old, runs, meas = None, 0, None
    if os.path.exists(a.out):
        text = open(a.out).read()
        old, runs, meas = parse_table(text), parse_runs(text), parse_measured(text)
    new = [res.get(l) for l in range(a.layers)]
    merged = merge_table(old, new, a.alpha, meas)
    if meas is not None and old is not None and len(old) == len(new):   # layers measured in earlier runs stay marked as measured
        merged = [m if (new[l] is not None or l in meas) else None for l, m in enumerate(merged)]
    tmp = a.out + ".tmp"
    open(tmp, "w").write(format_table(merged, runs + 1, "last part %s, %d layers measured, %.0f s" % (a.part, len(res), time.time() - t0)))
    os.replace(tmp, a.out)   # the engine sees the whole new file or the old one
    print("wrote %s (%d layers measured of %d)" % (a.out, len(res), a.layers))


if __name__ == "__main__":
    main()
