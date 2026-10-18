"""Predictor arms on the router trace (Qwen3.8-Flash-Next IQ3_S, our own engine, exact routing).
Target: layer l's true routing, from the router INPUT of layer l-d (d = 1, 2, 3), per token.
Arms: A router-reuse (layer l's real router on x_{l-d}, no training), B ridge regression onto layer l's router logits (trained here), and the same
for a held-out split.  Splits: IN = every 5th block of 40 windows held out (same prompts, other positions); SHIFT = train on the first 77% of the trace
(code, agent, reasoning, prose), test on the last 23% (other languages, summaries).
Metric: recall of the true rank 0-1 experts inside the predicted top-K (K = 2, 4, 8), ranks 0-3 inside top-8, and top-10 coverage."""
import struct, sys, time
import numpy as np
import torch

torch.set_num_threads(10)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace1.bin"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

# ---- load: a header pass finds the complete steps, a second pass copies rows straight into X[token, layer, :] (no intermediate copies)
buf = np.memmap(TRACE, dtype=np.uint8, mode="r")
off, recs = 4, []
while off + 8 <= len(buf):
    step, layer, nt, k = struct.unpack_from("<IHBB", buf, off)
    rec = 8 + nt * (H * 2 + k * 2)
    if off + rec > len(buf):
        break
    recs.append((step, layer, nt, off, k))
    off += rec
steps = {}
for step, layer, nt, o, k in recs:
    steps.setdefault(step, []).append((layer, nt, o, k))
complete = sorted(s for s, v in steps.items() if len(v) == NL and len({nt for _, nt, _, _ in v}) == 1)
N = sum(steps[s][0][1] for s in complete)
log("records", len(recs), "| complete steps", len(complete), "| tokens", N)
X = np.zeros((N, NL, H), np.uint16)
IDS = np.zeros((N, NL, 10), np.int16)
STEP_OF = np.zeros(N, np.int64)
tok = 0
for s in complete:
    for layer, nt, o, k in steps[s]:
        p = o + 8
        for t in range(nt):
            X[tok + t, layer] = np.frombuffer(buf[p:p + H * 2], dtype=np.uint16); p += H * 2
            IDS[tok + t, layer] = np.frombuffer(buf[p:p + k * 2], dtype=np.int16)[:10]; p += k * 2
    STEP_OF[tok:tok + steps[s][0][1]] = s
    tok += steps[s][0][1]
log("loaded X %.1f GB" % (X.nbytes / 1e9))
W = torch.from_numpy(np.load("/mnt/nvme/strata-data/traces/routers_f32.npy"))      # [48, 512, 2560]
f32 = lambda u: torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))

# ---- splits
blk = (STEP_OF - STEP_OF.min()) // 40
te_in = (blk % 5 == 4)
cut = int(0.77 * N)
te_shift = np.arange(N) >= cut
splits = {"IN": (np.where(~te_in)[0], np.where(te_in)[0]), "SHIFT": (np.where(~te_shift)[0], np.where(te_shift)[0])}
log({k: (len(a), len(b)) for k, (a, b) in splits.items()})


def metrics(score, ids, tag):
    """score [n,512] float, ids [n,10] int (true rank order)."""
    top = score.topk(10, dim=1).indices.numpy()
    ids = ids.astype(np.int64)
    out = {}
    for K in (2, 4, 8):
        pk = top[:, :K]
        out[f"r01@{K}"] = float((ids[:, :2, None] == pk[:, None, :]).any(2).mean())
    out["r03@8"] = float((ids[:, :4, None] == top[:, None, :8]).any(2).mean())
    out["cov10"] = float(np.mean([(len(set(a) & set(b)) / 10.0) for a, b in zip(top.tolist(), ids.tolist())]))
    return out


LAMS = (0.3, 3.0)
results = {}
layers = list(range(1, 48))
for split, (tr, te) in splits.items():
    acc = {}
    for lin in range(0, 47):                                    # the INPUT layer; targets are lin + d
        Xi = f32(X[tr, lin]) if False else None
        Xtr = f32(X[tr, lin]).reshape(len(tr), H)
        mu = Xtr.mean(0, keepdim=True)
        Xc = Xtr - mu
        G = Xc.T @ Xc / len(tr)                                  # [H,H]
        lam_scale = float(G.diagonal().mean())
        Xte = f32(X[te, lin]).reshape(len(te), H)
        for d in (1, 2, 3):
            l = lin + d
            if l >= NL:
                continue
            Wl = W[l]
            Ytr = f32(X[tr, l]).reshape(len(tr), H) @ Wl.T       # teacher logits at layer l  [n,512]
            Yte_ids = IDS[te, l]
            ymu = Ytr.mean(0, keepdim=True)
            C = Xc.T @ (Ytr - ymu) / len(tr)                     # [H,512]
            best = None
            for lam in LAMS:
                Wr = torch.linalg.solve(G + lam * lam_scale * torch.eye(H), C)
                sc = (Xte - mu) @ Wr + ymu
                m = metrics(sc, Yte_ids, "ridge")
                if best is None or m["cov10"] > best[1]["cov10"]:
                    best = (lam, m)
            mA = metrics(Xte @ Wl.T, Yte_ids, "reuse")
            acc.setdefault(d, []).append((mA, best[1], best[0]))
    results[split] = acc
    log(split, "done")
    for d in (1, 2, 3):
        rows = acc[d]
        ma = {k: np.mean([r[0][k] for r in rows]) * 100 for k in rows[0][0]}
        mb = {k: np.mean([r[1][k] for r in rows]) * 100 for k in rows[0][1]}
        lams = [r[2] for r in rows]
        print(f"\n{split} | d={d} | layers scored {len(rows)} | ridge lambda chosen: {np.mean(np.array(lams) == LAMS[0]):.0%} at {LAMS[0]}")
        print(f"   {'':14s} {'r01@2':>7} {'r01@4':>7} {'r01@8':>7} {'r03@8':>7} {'cov10':>7}")
        print(f"   {'router-reuse':14s} " + " ".join(f"{ma[k]:7.1f}" for k in ('r01@2', 'r01@4', 'r01@8', 'r03@8', 'cov10')))
        print(f"   {'ridge (ours)':14s} " + " ".join(f"{mb[k]:7.1f}" for k in ('r01@2', 'r01@4', 'r01@8', 'r03@8', 'cov10')), flush=True)
import pickle
pickle.dump({s: {d: [(a, b, c) for a, b, c in v] for d, v in r.items()} for s, r in results.items()}, open("/mnt/nvme/strata-data/traces/arms_results.pkl", "wb"))
log("saved")
