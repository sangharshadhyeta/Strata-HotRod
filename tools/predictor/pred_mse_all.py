"""MSE-loss cheap learners: stage A (lr grid, hinge margin, top-K), stage B (all layers), stage C (published base at d=1).
Prequential as in pred_online.py: every token is scored with the current weights, then learned from (mini-batches of 16).
Learners start at the real router (reuse base) or at the published linear predictor (pub base, d=1 only).
Loss (name/mse): centred MSE on the true top-K logits + hinge keeping the other E-K logits below the K-th true logit minus margin.
Floors: lo in (2,3,4,6) = true ranks 0..lo-1 inside the predicted top-2lo.  Non-resident metrics p1/p2/p4/mr4 as in pred_online.py.
Trace: trace_tok.bin (the only trace on disk; see the report).  Prefill is not in the trace.
Usage: python pred_mse_all.py ABC   (stages to run; each writes its own output file)
"""
import struct, sys, time
import numpy as np
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace_tok.bin"
ROUTER = "/mnt/nvme/strata-data/traces/routers_f32.npy"
PUB = "/home/Projects/Strata/data/expert-predictor/la_model.bin"
OUTDIR = "/home/Projects/Strata-train/tools/predictor"
STAGES = sys.argv[1] if len(sys.argv) > 1 else "ABC"
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
log("loaded", N, "tokens from", TRACE, "all", NL, "layers")


def f32(u):
    return torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))


def load_pub(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"LAM1", "bad magic"
    nl, ne, hid = struct.unpack_from("<III", raw, 4)
    assert (nl, ne, hid) == (NL, E, H), (nl, ne, hid)
    off, out = 16, {}
    for l in range(nl):
        present = raw[off]
        off += 1
        if not present:
            continue
        wb = np.frombuffer(raw, dtype=np.uint16, count=ne * hid, offset=off).reshape(ne, hid)
        off += ne * hid * 2
        bb = np.frombuffer(raw, dtype=np.float32, count=ne, offset=off).copy()
        off += ne * 4
        out[l] = (f32(wb.copy()), torch.from_numpy(bb))
    return out


PUBW = load_pub(PUB)
log("published predictor layers:", sorted(PUBW)[:3], "...", len(PUBW), "layers")

_REC = {}


def REC(l):
    """RECENT[t, e]: expert e was routed at layer l within tokens t-32 .. t-1 (never-used = not recent)."""
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


def prep(d, l):
    return f32(X[l - d]), f32(X[l]) @ W[l].T, torch.from_numpy(IDS[l].astype(np.int64))


B = 16
SHIFT_AT = int(0.77 * N)
SWEEP_N = 3000
D_LIST = (2, 1)
LOS = (2, 3, 4, 6)
FLOORS = [f"lo{lo}" for lo in LOS]
NRM = ["p1", "p2", "p4", "mr4"]
METRICS = FLOORS + NRM
LAYERS5 = [8, 16, 24, 32, 40]
DEF_CFG = (0.1, 32)
MSE_GRID = [("lowrank-8/mse", "lowrank", [1e-4, 1e-3, 1e-2], 8),
            ("lowrank-32/mse", "lowrank", [1e-4, 1e-3, 1e-2], 32),
            ("bias/mse", "bias", [1e-3, 1e-2, 1e-1, 3e-1, 9e-1], 0),
            ("scale+bias/mse", "scale", [1e-4, 1e-3, 1e-2, 3e-2, 9e-2], 0)]
NAMES_MSE = [n for n, _, _, _ in MSE_GRID]


def metrics(logit, ids, rec):
    ids = ids.long()
    top = logit.topk(2 * max(LOS), dim=1).indices
    res = {}
    for lo in LOS:
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
    """Additive logit term: lowrank (A, B) or bias (b) or scale (s, b); zero (or s = 1) at init."""
    def __init__(self, kind, r=0):
        self.kind = kind
        if kind == "lowrank":
            self.A = (torch.randn(r, H) * 1e-3).requires_grad_(True)
            self.B = torch.zeros(E, r, requires_grad=True)
            self.params = [self.A, self.B]
        elif kind == "bias":
            self.b = torch.zeros(E, requires_grad=True)
            self.params = [self.b]
        else:
            self.s = torch.ones(E, requires_grad=True)
            self.b = torch.zeros(E, requires_grad=True)
            self.params = [self.s, self.b]

    def delta(self, x, base):
        if self.kind == "lowrank":
            return (x @ self.A.T) @ self.B.T
        if self.kind == "bias":
            return self.b
        return (self.s - 1) * base + self.b


def mse_loss(pred, Y, margin, K):
    v, i = Y.topk(K, dim=1)
    thr = v[:, -1:] - margin
    pc = pred.gather(1, i)
    dev = (pc - pc.mean(1, keepdim=True)) - (v - v.mean(1, keepdim=True))
    mask = torch.ones_like(pred).scatter_(1, i, 0.0)
    hinge = (torch.relu(pred - thr) * mask).sum(1) / (E - K)
    return (dev ** 2).mean() + hinge.mean()


class Learner:
    """specs: list of (kind, lr, r, (margin, K)).  base: None = the real router W0, or (Wb, bb) = a fixed linear base.  No parts = fixed base."""
    def __init__(self, W0, specs, base=None):
        self.W0, self.base = W0, base
        self.cfg = specs[0][3] if specs else None
        self.parts = [Part(kind, r) for kind, lr, r, cfg in specs]
        self.opts = [torch.optim.Adam(p.params, lr=lr) for p, (kind, lr, r, cfg) in zip(self.parts, specs)]
        self.n_params = sum(q.numel() for p in self.parts for q in p.params)

    def base_logits(self, x):
        if self.base is None:
            return x @ self.W0.T
        return x @ self.base[0].T + self.base[1]

    def logits(self, x):
        base = self.base_logits(x)
        out = base
        for p in self.parts:
            out = out + p.delta(x, base)
        return out

    def learn(self, x, Y):
        if not self.parts:
            return
        loss = mse_loss(self.logits(x), Y, self.cfg[0], self.cfg[1])
        for o in self.opts:
            o.zero_grad()
        loss.backward()
        for o in self.opts:
            o.step()


def simulate(xi, Y, ids, Wl, rec_all, arms, n_end):
    """arms: {name: (specs, base)}.  Returns per-token metrics per arm (reuse = the real router, never learns)."""
    L = {name: Learner(Wl, sp, base) for name, (sp, base) in arms.items()}
    names = ["reuse", *L]
    acc = {k: {m: [] for m in METRICS} for k in names}
    for i in range(0, n_end, B):
        x, y, idr, rec = xi[i:i + B], Y[i:i + B], ids[i:i + B], rec_all[i:i + B]
        with torch.no_grad():
            outs = {"reuse": x @ Wl.T, **{k: v.logits(x) for k, v in L.items()}}
            for k, lg in outs.items():
                mm = metrics(lg, idr, rec)
                for m in METRICS:
                    acc[k][m].append(mm[m])
        for v in L.values():
            v.learn(x, y)
    return {k: {m: torch.cat(acc[k][m]).numpy() for m in METRICS} for k in names}, {n: L[n].n_params for n in L}


def spec(kind, lr, r, cfg):
    return [(kind, lr, r, cfg)]


def window_row(res, a, b, metric):
    return np.mean([np.nanmean(res[metric][a:b]) for res in [res]]) * 100


def avg_table(F, title, rows, d, layers):
    """rows: list of (label, {layer: res}).  Prints first 2000 | before shift | after shift, mean over layers, for every metric."""
    F(f"\n=== {title} (d={d}) ===")
    for m in METRICS:
        lab = {"lo2": "lo2 true ranks 0-1 in top-4", "lo3": "lo3 true ranks 0-2 in top-6", "lo4": "lo4 true ranks 0-3 in top-8",
               "lo6": "lo6 true ranks 0-5 in top-12", "p1": "non-resident precision@1", "p2": "non-resident precision@2",
               "p4": "non-resident precision@4", "mr4": "non-resident miss-recall@4"}[m]
        F(f"--- {lab} (%): first 2000 | before shift | after shift")
        for label, per in rows:
            def av(a, b):
                return np.mean([np.nanmean(per[l][m][a:b]) for l in layers]) * 100
            F(f"   {label:24s} {av(0, 2000):6.1f} | {av(2000, SHIFT_AT):6.1f} | {av(SHIFT_AT, N):6.1f}")


F = None
def emit(*a):
    print(*a, file=F, flush=True)


# ================= stage A: lr grid, hinge margin, top-K (layers 8,16,24,32,40; first 3000 tokens for the sweeps)
SET = {}
if "A" in STAGES:
    F = open(f"{OUTDIR}/online_cheap_lr.out", "w")
    emit("STAGE A: lr grid widened once past any edge (bias/mse, scale+bias/mse), then margin {0.1,0.5} x K {16,32,64} for the best learner")
    for d in D_LIST:
        arms = {}
        for name, kind, lrs, r in MSE_GRID:
            for lr in lrs:
                arms[f"{name}@{lr:g}"] = (spec(kind, lr, r, DEF_CFG), None)
        sw = {k: [] for k in arms}
        for l in LAYERS5:
            xi, Y, ids = prep(d, l)
            res, _ = simulate(xi, Y, ids, W[l], REC(l), arms, SWEEP_N)
            for k in arms:
                sw[k].append(np.mean([res[k][m].mean() for m in FLOORS]) * 100)
        emit(f"\n=== lr sweep d={d}, first {SWEEP_N} tokens, mean over layers {LAYERS5} and floors (%) ===")
        SET[d] = {}
        for name, kind, lrs, r in MSE_GRID:
            vals = {lr: np.mean(sw[f"{name}@{lr:g}"]) for lr in lrs}
            for lr in lrs:
                emit(f"   {name:16s} lr={lr:<8g} {vals[lr]:6.2f}")
            ch = max(lrs, key=lambda lr: vals[lr])
            edge = " (EDGE of the widened grid)" if ch in (lrs[0], lrs[-1]) else ""
            emit(f"   -> chosen {name} lr={ch:g}{edge}")
            SET[d][name] = {"kind": kind, "r": r, "lr": ch, "cfg": DEF_CFG, "crit": vals[ch]}
    best_name = max(NAMES_MSE, key=lambda n: np.mean([SET[d][n]["crit"] for d in D_LIST]))
    emit(f"\nbest learner (mean of its d=1 and d=2 sweep criterion): {best_name}")
    for d in D_LIST:
        s = SET[d][best_name]
        arms = {}
        for m in (0.1, 0.5):
            for K in (16, 32, 64):
                arms[f"cfg margin={m} K={K}"] = (spec(s["kind"], s["lr"], s["r"], (m, K)), None)
        sw = {k: [] for k in arms}
        for l in LAYERS5:
            xi, Y, ids = prep(d, l)
            res, _ = simulate(xi, Y, ids, W[l], REC(l), arms, SWEEP_N)
            for k in arms:
                sw[k].append(np.mean([res[k][m].mean() for m in FLOORS]) * 100)
        emit(f"\n=== {best_name} hinge margin x top-K sweep, d={d}, lr={s['lr']:g}, first {SWEEP_N} tokens (mean floors %) ===")
        for k in arms:
            emit(f"   {k:24s} {np.mean(sw[k]):6.2f}")
        cbest = max(arms, key=lambda k: np.mean(sw[k]))
        m, K = cbest.split()[1].split("=")[1], cbest.split()[2].split("=")[1]
        SET[d][best_name]["cfg"] = (float(m), int(K))
        emit(f"   -> chosen cfg margin={m} K={K}")
    # full stream on the 5 layers with the chosen settings
    for d in D_LIST:
        arms = {n: (spec(SET[d][n]["kind"], SET[d][n]["lr"], SET[d][n]["r"], SET[d][n]["cfg"]), None) for n in NAMES_MSE}
        per = {}
        for l in LAYERS5:
            xi, Y, ids = prep(d, l)
            res, npar = simulate(xi, Y, ids, W[l], REC(l), arms, N)
            per[l] = res
        rows = [("reuse", per)] + [(n, per) for n in NAMES_MSE]
        # per-learner rows: the same layer results, one row per learner
        emit(f"\n=== FULL STREAM, layers {LAYERS5}, chosen settings: " +
             ", ".join(f"{n} lr={SET[d][n]['lr']:g} cfg={SET[d][n]['cfg']}" for n in NAMES_MSE) + f" (d={d}) ===")
        for m in METRICS:
            emit(f"--- d={d} {m}: first 2000 | before shift | after shift (%)")
            for label in ["reuse"] + NAMES_MSE:
                emit(f"   {label:16s} " + " | ".join(f"{np.mean([np.nanmean(per[l][label][m][a:b]) for l in LAYERS5]) * 100:6.1f}"
                                                   for a, b in ((0, 2000), (2000, SHIFT_AT), (SHIFT_AT, N))))
        log(f"stage A d={d} full stream done")
    F.close()
    log("stage A written to", f"{OUTDIR}/online_cheap_lr.out")

# ================= stage B: all layers the lead allows, with the stage-A settings
RES_B = {}
if "B" in STAGES:
    if not SET:
        raise SystemExit("stage B needs stage A's settings: run ABC or AB")
    F = open(f"{OUTDIR}/online_cheap_all.out", "w")
    emit("STAGE B: all layers (d=1: layers 1-47; d=2: layers 2-47) with the stage-A settings; reuse = the real router")
    for d in D_LIST:
        arms = {n: (spec(SET[d][n]["kind"], SET[d][n]["lr"], SET[d][n]["r"], SET[d][n]["cfg"]), None) for n in NAMES_MSE}
        layers = list(range(d, NL))
        for l in layers:
            xi, Y, ids = prep(d, l)
            res, _ = simulate(xi, Y, ids, W[l], REC(l), arms, N)
            RES_B[(d, l)] = res
        log(f"stage B d={d} done ({len(layers)} layers)")
        rows = [("reuse", {l: RES_B[(d, l)] for l in layers})] + [(n, {l: {"reuse": RES_B[(d, l)]["reuse"], n: RES_B[(d, l)][n]} for l in layers}) for n in NAMES_MSE]
        # build one row per learner with its own metrics
        for m in METRICS:
            emit(f"\n=== STAGE B d={d} {m}: mean over layers {layers[0]}-{layers[-1]} (%): first 2000 | before shift | after shift ===")
            for label in ["reuse"] + NAMES_MSE:
                emit(f"   {label:16s} " + " | ".join(
                    f"{np.mean([np.nanmean(RES_B[(d, l)][label][m][a:b]) for l in layers]) * 100:6.1f}" for a, b in ((0, 2000), (2000, SHIFT_AT), (SHIFT_AT, N))))
        emit(f"\n--- d={d} per-layer lo3 full stream (%) ---")
        emit("layer   reuse  " + "  ".join(f"{n[:12]:>12s}" for n in NAMES_MSE))
        lo3 = {}
        for l in layers:
            vals = {lab: np.mean(RES_B[(d, l)][lab]["lo3"]) * 100 for lab in ["reuse"] + NAMES_MSE}
            lo3[l] = vals
            emit(f"{l:5d}  {vals['reuse']:6.1f}  " + "  ".join(f"{vals[n]:12.1f}" for n in NAMES_MSE))
        worst = sorted(layers, key=lambda l: lo3[l][NAMES_MSE[0]])[:5]
        emit(f"5 worst layers for {NAMES_MSE[0]} (lo3 full stream): " + ", ".join(f"{l} ({lo3[l][NAMES_MSE[0]]:.1f}, reuse {lo3[l]['reuse']:.1f})" for l in worst))
        worst_b = sorted(layers, key=lambda l: lo3[l]["reuse"])[:5]
        emit(f"5 worst layers for reuse (lo3 full stream): " + ", ".join(f"{l} ({lo3[l]['reuse']:.1f})" for l in worst_b))
    F.close()
    log("stage B written")

# ================= stage C: published linear predictor as the base (d=1 only), all layers 1-47
if "C" in STAGES:
    if not SET:
        raise SystemExit("stage C needs stage A's settings")
    F = open(f"{OUTDIR}/online_pub.out", "w")
    d = 1
    emit("STAGE C: d=1, all layers 1-47. pub = published linear predictor (fixed, folded bf16, applied to x_{l-1}); reuse = real router")
    res_c = {}
    for l in range(1, NL):
        Wp = PUBW[l]
        arms = {
            "pub (fixed)": ([], Wp),
            "pub+lowrank-8/mse": (spec("lowrank", SET[d]["lowrank-8/mse"]["lr"], 8, SET[d]["lowrank-8/mse"]["cfg"]), Wp),
            "pub+bias/mse": (spec("bias", SET[d]["bias/mse"]["lr"], 0, SET[d]["bias/mse"]["cfg"]), Wp),
        }
        xi, Y, ids = prep(d, l)
        res, _ = simulate(xi, Y, ids, W[l], REC(l), arms, N)
        res_c[l] = res
        # the reuse arms from stage B's d=1 run (same settings)
        res_c[l]["reuse+lowrank-8/mse"] = RES_B[(d, l)]["lowrank-8/mse"]
        res_c[l]["reuse+bias/mse"] = RES_B[(d, l)]["bias/mse"]
    log("stage C done")
    layers = list(range(1, NL))
    labels = ["reuse", "reuse+lowrank-8/mse", "reuse+bias/mse", "pub (fixed)", "pub+lowrank-8/mse", "pub+bias/mse"]
    for m in METRICS:
        emit(f"\n=== STAGE C d=1 {m}: mean over layers 1-47 (%): first 2000 | before shift | after shift ===")
        for label in labels:
            emit(f"   {label:20s} " + " | ".join(
                f"{np.mean([np.nanmean(res_c[l][label][m][a:b]) for l in layers]) * 100:6.1f}" for a, b in ((0, 2000), (2000, SHIFT_AT), (SHIFT_AT, N))))
    emit(f"\n--- d=1 per-layer lo3 full stream (%) ---")
    emit("layer  " + " ".join(f"{lab[:14]:>14s}" for lab in labels))
    for l in layers:
        emit(f"{l:5d}  " + " ".join(f"{np.mean(res_c[l][lab]['lo3']) * 100:14.1f}" for lab in labels))
    F.close()
    log("stage C written")
