"""STEP E: longer leads on trace_tok.bin.  Head = lowrank-8/mse (lr 1e-3, MSE settings per d: d=1 (0.1,16); d>=2 (0.1,64)), reuse = router.
Heads at d = 3, 4, 6 for all target layers the lead allows (l >= d), plus d = 1 and 2 (needed for the cascade).
Cascade (target layer l, l >= 4): the union of the d=4 head's top-3, the d=2 head's top-3 and the d=1 head's top-3, scored on the true
routed ranks 0-2 of layer l: all ranks (catch) and the non-resident subset (true ranks 0-2 not routed at layer l within the last 32 tokens).
Also the distinct guesses per token (union size), and those that are non-recent.
Output: tools/predictor/online_lead.out
"""
import struct, time
import numpy as np
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace_tok.bin"
ROUTER = "/mnt/nvme/strata-data/traces/routers_f32.npy"
OUT = "/home/Projects/Strata-train/tools/predictor/online_lead.out"
SET = {1: (1e-3, (0.1, 16)), 2: (1e-3, (0.1, 64)), 3: (1e-3, (0.1, 64)), 4: (1e-3, (0.1, 64)), 6: (1e-3, (0.1, 64))}
HEADS = [1, 2, 3, 4, 6]
B = 16
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

buf = np.memmap(TRACE, dtype=np.uint8, mode="r")
off, recs = 4, []
while off + 8 <= len(buf):
    step, layer, nt, k = struct.unpack_from("<IHBB", buf, off)
    if layer == 0xFFFF:
        off += 8 + nt * 4
        continue
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
X = {l: np.zeros((N, H), np.uint16) for l in range(NL)}
IDS = {l: np.zeros((N, 10), np.int16) for l in range(NL)}
tok = 0
for s in complete:
    for layer, nt, o, k in steps[s]:
        region = np.asarray(buf[o + 8:o + 8 + nt * (H * 2 + k * 2)]).reshape(nt, H * 2 + k * 2)
        X[layer][tok:tok + nt] = np.ascontiguousarray(region[:, :H * 2]).view(np.uint16)
        IDS[layer][tok:tok + nt] = np.ascontiguousarray(region[:, H * 2:H * 2 + k * 2]).view(np.int16)[:, :10]
    tok += steps[s][0][1]
W = torch.from_numpy(np.load(ROUTER))
f32 = lambda u: torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))
SHIFT_AT = int(0.77 * N)
log("loaded", N, "tokens")

_REC = {}


def REC(l):
    if l not in _REC:
        ids_np = IDS[l].astype(np.int64)
        R = np.zeros((N, E), bool)
        cnt = np.zeros(E, np.int64)
        for t in range(N):
            R[t] = cnt > 0
            cnt[ids_np[t]] += 1
            if t >= 32:
                cnt[ids_np[t - 32]] -= 1
        _REC[l] = torch.from_numpy(R)
    return _REC[l]


def metrics(logit, ids, rec):
    ids = ids.long()
    top = logit.topk(12, dim=1).indices
    res = {}
    for lo in (2, 3, 6):
        res[f"lo{lo}"] = (ids[:, :lo, None] == top[:, None, :2 * lo]).any(2).float().mean(1)
    p = logit.masked_fill(rec, -1e9).topk(4, dim=1).indices
    pick = (p[:, :, None] == ids[:, None, :])
    nr_true = ~rec.gather(1, ids)
    res["mr4"] = (pick.any(1) & nr_true).float().sum(1) / nr_true.float().sum(1)
    return res


def mse_loss(pred, Y, margin, K):
    v, i = Y.topk(K, dim=1)
    thr = v[:, -1:] - margin
    pc = pred.gather(1, i)
    dev = (pc - pc.mean(1, keepdim=True)) - (v - v.mean(1, keepdim=True))
    mask = torch.ones_like(pred).scatter_(1, i, 0.0)
    hinge = (torch.relu(pred - thr) * mask).sum(1) / (E - K)
    return (dev ** 2).mean() + hinge.mean()


KEYS = ["lo2", "lo3", "lo6", "mr4"]


def run_head(d, l):
    lr, cfg = SET[d]
    xi = f32(X[l - d])
    Y = f32(X[l]) @ W[l].T
    ids = torch.from_numpy(IDS[l].astype(np.int64))
    rec = REC(l)
    Wl = W[l]
    A = (torch.randn(8, H) * 1e-3).requires_grad_(True)
    Bm = torch.zeros(E, 8, requires_grad=True)
    opt = torch.optim.Adam([A, Bm], lr=lr)
    acc = {a: {k: [] for k in KEYS} for a in ("reuse", "lr")}
    top3 = []
    for i in range(0, N, B):
        x, y, idr, rb = xi[i:i + B], Y[i:i + B], ids[i:i + B], rec[i:i + B]
        with torch.no_grad():
            lg_r = x @ Wl.T
            lg_l = lg_r + (x @ A.T) @ Bm.T
            for a, lg in (("reuse", lg_r), ("lr", lg_l)):
                mm = metrics(lg, idr, rb)
                for k in KEYS:
                    acc[a][k].append(mm[k])
            top3.append(lg_l.topk(3, dim=1).indices)
        loss = mse_loss(lg_r + (x @ A.T) @ Bm.T, y, cfg[0], cfg[1])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = {a: {k: torch.cat(acc[a][k]).numpy() for k in KEYS} for a in acc}
    out["top3"] = torch.cat(top3)
    return out


RES = {}
for d in HEADS:
    for l in range(d, NL):
        RES[(d, l)] = run_head(d, l)
    log(f"head d={d} done")

with open(OUT, "w") as F:
    def emit(*a):
        print(*a, file=F, flush=True)
    emit(f"STEP E: longer leads, trace_tok.bin ({N} decode tokens), lowrank-8/mse lr 1e-3, MSE settings per d "
         f"(d=1: margin 0.1 K 16; d>=2: margin 0.1 K 64). Targets: every layer the lead allows.")
    emit("NOTE: d=3, 4, 6 use the K=64 setting (the d=2 choice); not tuned for these leads.")
    for d in (3, 4, 6):
        layers = list(range(d, NL))
        emit(f"\n=== d={d}: mean over layers {layers[0]}-{layers[-1]} (%): first 2000 | before shift | after shift ===")
        for k in KEYS:
            emit(f"--- {k}")
            for a in ("reuse", "lr"):
                vals = [np.mean([np.nanmean(RES[(d, l)][a][k][s:e]) for l in layers]) * 100 for s, e in ((0, 2000), (2000, SHIFT_AT), (SHIFT_AT, N))]
                lab = "reuse" if a == "reuse" else "lowrank-8/mse"
                emit(f"   {lab:14s} " + " | ".join(f"{v:6.1f}" for v in vals))

    # cascade: target layers l >= 4, heads 1, 2, 4
    cl = list(range(4, NL))
    emit(f"\n=== cascade at lo3 (target layers {cl[0]}-{cl[-1]}; heads lowrank-8/mse, top-3 each) ===")
    emit("catch = share of true ranks 0-2 in the set; NR = same over true ranks 0-2 not routed at the target layer within 32 tokens")
    for name, heads in (("d=4 head alone", [4]), ("d=2 head alone", [2]), ("d=1 head alone", [1]), ("union d=4+d=2+d=1", [4, 2, 1])):
        stats = {"catch": [], "NR": [], "distinct": [], "distinct_nonrecent": []}
        for l in cl:
            ids = torch.from_numpy(IDS[l].astype(np.int64))
            rec = REC(l)
            mask = torch.zeros(N, E)
            for d in heads:
                mask.scatter_(1, RES[(d, l)]["top3"], 1.0)
            inset = mask > 0
            ids3 = ids[:, :3]
            nr3 = ~rec.gather(1, ids3)
            stats["catch"].append(inset.gather(1, ids3).float().mean(1).numpy())
            captured = (inset.gather(1, ids3) & nr3).float().sum(1)
            stats["NR"].append((captured / nr3.float().sum(1)).numpy())
            stats["distinct"].append(inset.float().sum(1).numpy())
            stats["distinct_nonrecent"].append((inset & ~rec).float().sum(1).numpy())
        def seg(key, s, e):
            return np.mean([np.nanmean(v[s:e]) for v in stats[key]]) * 100
        if name.startswith("union"):
            pass
        emit(f"--- {name}")
        for key in ("catch", "NR"):
            emit(f"   {key:8s} " + " | ".join(f"{seg(key, s, e):6.1f}" for s, e in ((0, 2000), (2000, SHIFT_AT), (SHIFT_AT, N))) + "  (first 2000 | before | after, %)")
        for key in ("distinct", "distinct_nonrecent"):
            emit(f"   {key:18s} " + " | ".join(f"{np.mean([np.nanmean(v[s:e]) for v in stats[key]]):6.2f}" for s, e in ((0, 2000), (2000, SHIFT_AT), (SHIFT_AT, N)))
                 + "  (guesses per token per layer)")
    log("written", OUT)
