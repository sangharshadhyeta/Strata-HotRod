"""STEP D: prefill topic test on trace-prefill.bin (RTR2).  Layers 8,16,24,32,40; d=1 and d=2; learner = lowrank-8/mse (stage-A settings).
Requests are processed in file order; request i (1-based) belongs to conversation CONV_OF[i] (trace_prompts.PREFILL turn counts).
Arms (all scored on the request's DECODE tokens before learning on them; decode is exact in this trace):
  reuse     the plain router
  b         global learner, trained online on decode only
  c         global learner, trained on each request's prefill rows (every 4th prompt token) before that request's decode, then online on decode
  d         c plus a per-request part (lowrank-4), reset to zero at each request start, trained on that request's prefill and decode
  e         global learner trained on requests 1-18 (conversations 1-8, prefill + decode), frozen, scored on requests 19-24 (conversations 9-13)
Metrics: lo2 / lo3 / lo6 (true ranks 0-1 / 0-2 / 0-5 inside predicted top-4 / top-6 / top-12) and non-resident p1, p2, p4, mr4.
Residency proxy: as in pred_online.py, over the decode stream in request order.  Prefill is not scored.
Output: tools/predictor/online_prefill.out
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/Projects/Strata-train/tools/predictor")
from trace_reader import iter_records

torch.set_num_threads(4)
torch.manual_seed(0)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace-prefill.bin"
ROUTER = "/mnt/nvme/strata-data/traces/routers_f32.npy"
OUT = "/home/Projects/Strata-train/tools/predictor/online_prefill.out"
LAYERS5 = [8, 16, 24, 32, 40]
NEED = sorted({l - d for l in LAYERS5 for d in (1, 2)} | set(LAYERS5))
SET = {1: (1e-3, (0.1, 16)), 2: (1e-3, (0.1, 64))}              # stage A: lowrank-8/mse lr and (margin, K) per d
PER_RANK = 4
B = 16
LOS = (2, 3, 6)
FLOORS = [f"lo{lo}" for lo in LOS]
NRM = ["p1", "p2", "p4", "mr4"]
METRICS = FLOORS + NRM
SPLIT_REQ = 10   # the 24-request plan (requests 1-18 = conversations 1-8) is not in the trace: requests 1-10 train (e), 11-15 are scored
CONV_TURNS = [4, 2, 4, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1]            # trace_prompts.PREFILL turn counts, conversations a..m
CONV_OF = [c + 1 for c, n in enumerate(CONV_TURNS) for _ in range(n)]
CONV_LETTER = "abcdefghijklm"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

# ---- load (decode and prefill rows of the layers in NEED; request records give the request order)
dec, pre, reqs, cur = {}, {}, [], None
for r in iter_records(TRACE, want_x=True):
    k = r["kind"]
    if k == "request":
        cur = r["request"]
        reqs.append(cur)
        dec[cur], pre[cur] = {}, {}
    elif cur is None:
        continue
    elif k == "decode" and r["layer"] in NEED:
        dec[cur].setdefault(r["layer"], []).append((r["x"], r["ids"]))
    elif k == "prefill" and r["layer"] in NEED:
        pre[cur][r["layer"]] = (r["pos"], r["x"], r["ids"])
log(f"requests in file order: {len(reqs)} of {sum(CONV_TURNS)} planned (the trace is incomplete: its last write stopped after request {len(reqs)})", reqs)

Dx, Di, Px, Pi, Pp = {}, {}, {}, {}, {}
for i, q in enumerate(reqs):
    Dx[q], Di[q], Px[q], Pi[q], Pp[q] = {}, {}, {}, {}, {}
    lens = {l: sum(c[0].shape[0] for c in dec[q][l]) for l in NEED}
    assert len(set(lens.values())) == 1, (q, lens)
    for l in NEED:
        Dx[q][l] = np.concatenate([c[0] for c in dec[q][l]])
        Di[q][l] = np.concatenate([c[1] for c in dec[q][l]])[:, :10]
        pos, x, ids = pre[q][l]
        o = np.argsort(pos, kind="stable")
        Px[q][l], Pi[q][l], Pp[q][l] = x[o], ids[o][:, :10], pos[o]
    ps = {l: Pp[q][l].tolist() for l in NEED}
    assert all(ps[l] == ps[NEED[0]] for l in NEED), f"request {q}: prefill positions differ across layers"
W = torch.from_numpy(np.load(ROUTER))
f32 = lambda u: torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))
T = {q: Dx[q][NEED[0]].shape[0] for q in reqs}
NP = {q: Px[q][NEED[0]].shape[0] for q in reqs}
log("decode tokens per request:", [T[q] for q in reqs])
log("prefill rows per request:", [NP[q] for q in reqs])


def metrics(logit, ids, rec):
    ids = ids.long()
    top = logit.topk(12, dim=1).indices
    res = {}
    for lo in (2, 3, 6):
        res[f"lo{lo}"] = (ids[:, :lo, None] == top[:, None, :2 * lo]).any(2).float().mean(1)
    p = logit.masked_fill(rec, -1e9).topk(4, dim=1).indices
    pick = (p[:, :, None] == ids[:, None, :])
    hit = pick.any(2) & ~rec.gather(1, p)
    res["p1"] = hit[:, :1].float().mean(1)
    res["p2"] = hit[:, :2].float().sum(1) / 2
    res["p4"] = hit.float().sum(1) / 4
    nr_true = ~rec.gather(1, ids)
    res["mr4"] = (pick.any(1) & nr_true).float().sum(1) / nr_true.float().sum(1)
    return res


class Part:
    def __init__(self, kind, r):
        if kind == "lowrank":
            self.A = (torch.randn(r, H) * 1e-3).requires_grad_(True)
            self.B = torch.zeros(E, r, requires_grad=True)
            self.params = [self.A, self.B]

    def delta(self, x):
        return (x @ self.A.T) @ self.B.T


def mse_loss(pred, Y, margin, K):
    v, i = Y.topk(K, dim=1)
    thr = v[:, -1:] - margin
    pc = pred.gather(1, i)
    dev = (pc - pc.mean(1, keepdim=True)) - (v - v.mean(1, keepdim=True))
    mask = torch.ones_like(pred).scatter_(1, i, 0.0)
    hinge = (torch.relu(pred - thr) * mask).sum(1) / (E - K)
    return (dev ** 2).mean() + hinge.mean()


class Lrn:
    """Learner on the real router Wl: parts is a list of ranks (lowrank); the last part can be replaced per request."""
    def __init__(self, Wl, lr, cfg, ranks):
        self.Wl, self.lr, self.cfg = Wl, lr, cfg
        self.parts, self.opts = [], []
        for r in ranks:
            self.add(r)

    def add(self, r):
        p = Part("lowrank", r)
        self.parts.append(p)
        self.opts.append(torch.optim.Adam(p.params, lr=self.lr))

    def replace(self, idx, r):
        self.parts[idx] = Part("lowrank", r)
        self.opts[idx] = torch.optim.Adam(self.parts[idx].params, lr=self.lr)

    def logits(self, x):
        out = x @ self.Wl.T
        for p in self.parts:
            out = out + p.delta(x)
        return out

    def learn(self, x, Y):
        loss = mse_loss(self.logits(x), Y, self.cfg[0], self.cfg[1])
        for o in self.opts:
            o.zero_grad()
        loss.backward()
        for o in self.opts:
            o.step()


ARMS = ["reuse", "b", "c", "d", "e"]
RES = {}
for d in (1, 2):
    lr, cfg = SET[d]
    for l in LAYERS5:
        Wl = W[l]
        rec_stream = np.zeros((sum(T.values()), E), bool)
        ids_stream = np.concatenate([Di[q][l] for q in reqs]).astype(np.int64)
        cnt = np.zeros(E, np.int64)
        for t in range(ids_stream.shape[0]):
            rec_stream[t] = cnt > 0
            cnt[ids_stream[t]] += 1
            if t >= 32:
                cnt[ids_stream[t - 32]] -= 1
        rec_stream = torch.from_numpy(rec_stream)
        L = {"b": Lrn(Wl, lr, cfg, [8]), "c": Lrn(Wl, lr, cfg, [8]), "d": Lrn(Wl, lr, cfg, [8, PER_RANK]),
             "e": Lrn(Wl, lr, cfg, [8])}
        acc = {a: {m: [] for m in METRICS} for a in ARMS}
        pos_in_req, req_of = [], []
        off = 0
        for qi, q in enumerate(reqs):
            n_dec, n_pre = T[q], NP[q]
            xd, yd = f32(Dx[q][l - d]), f32(Dx[q][l]) @ Wl.T
            idd = torch.from_numpy(Di[q][l].astype(np.int64))
            xp, yp = f32(Px[q][l - d]), f32(Px[q][l]) @ Wl.T
            rec = rec_stream[off:off + n_dec]
            e_trains = qi < SPLIT_REQ
            # per-request part: fresh at each request start
            L["d"].replace(1, PER_RANK)
            # prefill training for c, d (and e while it is still training)
            for i in range(0, n_pre, B):
                xb, yb = xp[i:i + B], yp[i:i + B]
                L["c"].learn(xb, yb)
                L["d"].learn(xb, yb)
                if e_trains:
                    L["e"].learn(xb, yb)
            # decode: score every arm, then learn on the batch
            for i in range(0, n_dec, B):
                xb, idb, rb = xd[i:i + B], idd[i:i + B], rec[i:i + B]
                with torch.no_grad():
                    outs = {"reuse": xb @ Wl.T, **{a: L[a].logits(xb) for a in ("b", "c", "d", "e")}}
                    for a, lg in outs.items():
                        mm = metrics(lg, idb, rb)
                        for m in METRICS:
                            acc[a][m].append(mm[m])
                for a in ("b", "c", "d"):
                    L[a].learn(xb, yd[i:i + B])
                if e_trains:
                    L["e"].learn(xb, yd[i:i + B])
            pos_in_req.append(np.arange(n_dec))
            req_of.append(np.full(n_dec, qi))
            off += n_dec
        RES[(d, l)] = {a: {m: torch.cat(acc[a][m]).numpy() for m in METRICS} for a in ARMS}
        RES[(d, l)]["_pos"] = np.concatenate(pos_in_req)
        RES[(d, l)]["_req"] = np.concatenate(req_of)
        log(f"d={d} layer {l} done")

# ---- report
POS = None
with open(OUT, "w") as F:
    def emit(*a):
        print(*a, file=F, flush=True)
    emit(f"STEP D: prefill topic test, trace-prefill.bin, {len(reqs)} requests (of 24 planned; the trace stopped after request {len(reqs)}), "
         f"layers {LAYERS5}, lowrank-8/mse (stage-A settings per d), decode tokens scored before learning.")
    emit(f"(e) substitute split: requests 1-{SPLIT_REQ} train the global part and freeze it, requests {SPLIT_REQ + 1}-{len(reqs)} are scored. "
         "The planned conversation split (1-18 vs 19-24) cannot be used: the trace does not contain those requests.")
    emit("request -> conversation (planned order, not verified against the trace): " +
         ", ".join(f"{i+1}:{CONV_LETTER[CONV_OF[i]-1]}" for i in range(len(reqs))))
    names = {"reuse": "reuse (router)", "b": "b global, decode only", "c": "c global + prefill",
             "d": "d global + prefill + per-request r4", "e": f"e frozen global (1-{SPLIT_REQ} trained)"}

    def block(title, d, sel_fn, arms):
        emit(f"\n=== d={d} {title} ===")
        emit(f"{'':32s}" + " ".join(f"{m:>7s}" for m in METRICS))
        for a in arms:
            vals = []
            for m in METRICS:
                vals.append(np.mean([np.nanmean(RES[(d, l)][a][m][sel_fn(RES[(d, l)])]) for l in LAYERS5]) * 100)
            emit(f"   {names[a]:30s}" + " ".join(f"{v:7.1f}" for v in vals))

    for d in (1, 2):
        block("all decode tokens of requests 1-24 (mean over layers, %)", d, lambda R: np.ones(len(R["_pos"]), bool), ARMS[:4])
        block("first 100 decode tokens of each request, requests 1-24 (%)", d, lambda R: R["_pos"] < 100, ARMS[:4])
        block(f"requests {SPLIT_REQ + 1}-{len(reqs)}, all decode tokens: plain router vs frozen global (e) (%)", d,
              lambda R: R["_req"] >= SPLIT_REQ, ["reuse", "e"])
        block(f"requests {SPLIT_REQ + 1}-{len(reqs)}, first 100 decode tokens each (%)", d,
              lambda R: (R["_req"] >= SPLIT_REQ) & (R["_pos"] < 100), ["reuse", "e"])
        emit(f"\n--- d={d} per request, lo3 over all decode tokens (mean over layers, %) ---")
        emit(f"{'req conv':10s}" + " ".join(f"{a:>9s}" for a in ARMS))
        for qi in range(len(reqs)):
            row = []
            for a in ARMS:
                v = np.mean([np.nanmean(RES[(d, l)][a]["lo3"][RES[(d, l)]["_req"] == qi]) for l in LAYERS5]) * 100
                row.append(v)
            emit(f"{qi+1:3d} {CONV_LETTER[CONV_OF[qi]-1]:>5s}   " + " ".join(f"{v:9.1f}" for v in row))
    log("written", OUT)
