"""Live-learning simulation (prequential: every token is SCORED with the current weights first, then LEARNED from).
The trace is streamed in its original order: code -> agent -> reasoning -> prose -> other languages -> summaries, so the shift at ~77% is a real
topic change.  Learners (per layer l, lead d; input = router input of layer l-d, target = layer l's true routing); all start EXACTLY at the real router:
  reuse       layer l's real router on x_{l-d}; never learns (the baseline Strata has today)
  online-R    full router copy (512x2560 + bias), Adam on a listwise loss against the true routing weights
  lowrank-r   logits = W @ x + B @ (A @ x); W fixed; A (r x 2560, std 1e-3), B (512 x r, zeros); Adam on A and B
  bias        logits = W @ x + b; Adam on b (512)
  scale+bias  logits = s * (W @ x) + b; s (512, init 1), b (512, init 0); Adam on s and b
Each cheap learner also runs with the MSE loss (name/mse): centred MSE on the true top-32 logits + hinge keeping the other 480 below the
32nd true logit minus 0.1.  Composites on the best cheap global G (listwise): pp-G (G reset to the real router at each prompt start);
G+P (G persistent plus a second part P that resets at each prompt start, both trained; P = bias or lowrank-8).
Prompt starts come from the server log only if they match the trace exactly.
Floors: lo in (2,3,4,6) scores true ranks 0..lo-1 inside the predicted top-2lo.  lr sweeps and the best-cheap choice use the mean over floors.
Non-resident metrics: a residency proxy marks expert e of layer l "recent" if it was routed at layer l within the last 32 tokens (never-used = not
recent).  p1/p2/p4 = share of the top-k picks among non-recent experts that are truly routed (non-recent picks only); mr4 = share of the true
non-recent routed experts captured in the top-4 non-recent picks.
Engine realism: the label is the layer's routed top-10 (softmax over the 10 chosen logits for the listwise loss); updates in mini-batches of 16
tokens (a reset takes effect at the batch that holds the prompt start).  The prefill of each prompt is not in the trace (decode tokens only),
so learning from a prompt's own prefill is not simulated.
Usage: python pred_online.py [trace.bin]   (default: traces/trace1.bin; token-id records of newer traces are skipped)
"""
import struct, sys, time
import numpy as np
import torch

torch.set_num_threads(10)
torch.manual_seed(0)
H, NL, E = 2560, 48, 512
TRACE = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/strata-data/traces/trace1.bin"
SERVER_LOG = "/root/.claude/jobs/743230e4/tmp/trace_server.log"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

buf = np.memmap(TRACE, dtype=np.uint8, mode="r")
off, recs = 4, []
while off + 8 <= len(buf):
    step, layer, nt, k = struct.unpack_from("<IHBB", buf, off)
    if layer == 0xFFFF:                                  # token-id record (newer traces): u32 step | u16 0xFFFF | u8 n | u8 0 | i32[n]
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
LAYERS = [8, 16, 24, 32, 40]
need = sorted({l - d for l in LAYERS for d in (1, 2)} | set(LAYERS))
X = {l: np.zeros((N, H), np.uint16) for l in need}
IDS = {l: np.zeros((N, 10), np.int16) for l in LAYERS}
tok = 0
for s in complete:
    for layer, nt, o, k in steps[s]:
        if layer not in X:
            continue
        p = o + 8
        for t in range(nt):
            X[layer][tok + t] = np.frombuffer(buf[p:p + H * 2], dtype=np.uint16); p += H * 2
            if layer in IDS:
                IDS[layer][tok + t] = np.frombuffer(buf[p:p + k * 2], dtype=np.int16)[:10]
            p += k * 2
    tok += steps[s][0][1]
W = torch.from_numpy(np.load("/mnt/nvme/strata-data/traces/routers_f32.npy"))
f32 = lambda u: torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))
log("loaded", N, "tokens from", TRACE, "layers", need)


def prompt_starts():
    """Token index where each prompt starts, or None.  Uses the generated counts of the server log's last engine run; accepted only if
    the cumulative window sums hit those boundaries exactly and cover all N tokens."""
    lines = open(SERVER_LOG).read().splitlines()
    engine = [i for i, ln in enumerate(lines) if "engine started" in ln]
    if not engine:
        return None, "no engine-start line in the server log"
    gen = []
    for ln in lines[engine[-1]:]:
        w = ln.split()
        if "generated" in w:
            gen.append(int(w[w.index("generated") - 1]))
    bounds, acc = {0}, 0
    for s in complete:
        acc += steps[s][0][1]
        bounds.add(acc)
    for off_tok in (0, 1):                                # 1: the prompt's first token comes from prefill, not a decode window
        starts, c = [], 0
        for g in gen:
            starts.append(c)
            c += g - off_tok
        if c == N and all(b in bounds for b in starts):
            return starts, f"{len(gen)} requests, decode offset {off_tok}"
    return None, f"{len(gen)} logged requests do not match the trace's window boundaries"


def recent_masks():
    """RECENT[l][t, e]: expert e was routed at layer l within tokens t-32 .. t-1."""
    out = {}
    for l in LAYERS:
        ids_np = IDS[l].astype(np.int64)
        R = np.zeros((N, E), bool)
        cnt = np.zeros(E, np.int64)
        for t in range(N):
            R[t] = cnt > 0
            cnt[ids_np[t]] += 1
            if t >= 32:
                cnt[ids_np[t - 32]] -= 1
        out[l] = torch.from_numpy(R)
    return out


STARTS, STARTS_WHY = prompt_starts()
log("prompt starts:", STARTS_WHY)
RECENT = recent_masks()
log("residency proxy built; recent share of routed experts per layer:",
    {l: round(float(RECENT[l].float().mean()), 3) for l in LAYERS})

B = 16
SHIFT_AT = int(0.77 * N)
SWEEP_N = 3000
D_LIST = (2, 1)
LOS = (2, 3, 4, 6)
FLOORS = [f"lo{lo}" for lo in LOS]
NRM = ["p1", "p2", "p4", "mr4"]
CHEAP = [("lowrank-8", "lowrank", 8), ("lowrank-32", "lowrank", 32), ("bias", "bias", 0), ("scale+bias", "scale", 0)]
GRID_LIST = [("online-R", "full", [3e-5, 1e-4, 3e-4], 0),
             ("lowrank-8", "lowrank", [1e-4, 1e-3, 1e-2], 8),
             ("lowrank-32", "lowrank", [1e-4, 1e-3, 1e-2], 32),
             ("bias", "bias", [1e-3, 1e-2, 1e-1], 0),
             ("scale+bias", "scale", [1e-4, 1e-3, 1e-2], 0)]
GRID = [(n, k, lrs, r, "list") for n, k, lrs, r in GRID_LIST] + \
       [(n + "/mse", k, lrs, r, "mse") for n, k, lrs, r in GRID_LIST if k != "full"]


def metrics(logit, ids, rec):
    ids = ids.long()
    top = logit.topk(2 * max(LOS), dim=1).indices
    res = {}
    for lo in LOS:
        res[f"lo{lo}"] = (ids[:, :lo, None] == top[:, None, :2 * lo]).any(2).float().mean(1)
    p = logit.masked_fill(rec, -1e9).topk(4, dim=1).indices
    pick = (p[:, :, None] == ids[:, None, :])                                  # [T,4,10]
    hit = pick.any(2) & ~rec.gather(1, p)                                      # non-recent picks that are routed
    res["p1"] = hit[:, :1].float().mean(1)
    res["p2"] = hit[:, :2].float().sum(1) / 2
    res["p4"] = hit.float().sum(1) / 4
    nr_true = ~rec.gather(1, ids)
    capt = pick.any(1) & nr_true
    res["mr4"] = capt.float().sum(1) / nr_true.float().sum(1)                  # nan where no non-recent routed expert
    return res


class Part:
    """An additive logit term: lowrank (A, B), bias (b) or scale (s, b); zero (or s = 1) at init."""
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


def target(Y):                                                                    # routing weights: softmax over the chosen top-10 logits
    v, i = Y.topk(10, dim=1)
    p = torch.zeros_like(Y)
    p.scatter_(1, i, torch.softmax(v, dim=1))
    return p


def mse_loss(pred, Y):
    v, i = Y.topk(32, dim=1)
    thr = v[:, -1:] - 0.1
    pc = pred.gather(1, i)
    dev = (pc - pc.mean(1, keepdim=True)) - (v - v.mean(1, keepdim=True))
    mask = torch.ones_like(pred).scatter_(1, i, 0.0)
    hinge = (torch.relu(pred - thr) * mask).sum(1) / (E - 32)
    return (dev ** 2).mean() + hinge.mean()


class Learner:
    """specs: list of (kind, lr, r, per_prompt, loss); loss 'list' or 'mse'.  A single 'full' spec is the full 512x2560 router copy."""
    def __init__(self, W0, specs):
        self.W0, self.specs = W0, specs
        self.loss = specs[0][4]
        if specs[0][0] == "full":
            self.full = True
            self.W = W0.clone().requires_grad_(True)
            self.b = torch.zeros(E, requires_grad=True)
            self.opts = [torch.optim.Adam([self.W, self.b], lr=specs[0][1])]
            self.n_params = self.W.numel() + E
        else:
            self.full = False
            self.parts = [Part(kind, r) for kind, lr, r, per, loss in specs]
            self.opts = [torch.optim.Adam(p.params, lr=lr) for p, (kind, lr, r, per, loss) in zip(self.parts, specs)]
            self.n_params = sum(q.numel() for p in self.parts for q in p.params)

    def logits(self, x):
        if self.full:
            return x @ self.W.T + self.b
        base = x @ self.W0.T
        out = base
        for p in self.parts:
            out = out + p.delta(x, base)
        return out

    def learn(self, x, Y):
        lg = self.logits(x)
        loss = mse_loss(lg, Y) if self.loss == "mse" else -(target(Y) * torch.log_softmax(lg, dim=1)).sum(1).mean()
        for o in self.opts:
            o.zero_grad()
        loss.backward()
        for o in self.opts:
            o.step()

    def reset_per_prompt(self):
        if self.full:
            return
        for i, (kind, lr, r, per, loss) in enumerate(self.specs):
            if per:
                self.parts[i] = Part(kind, r)
                self.opts[i] = torch.optim.Adam(self.parts[i].params, lr=lr)


def simulate(xi, Y, ids, Wl, rec_all, specs, n_end, starts=None):
    """specs: {name: [(kind, lr, r, per_prompt, loss), ...]}.  Returns per-token metrics per learner and the trainable parameter counts."""
    L = {name: Learner(Wl, sp) for name, sp in specs.items()}
    names = ["reuse", *L]
    acc = {k: {m: [] for m in FLOORS + NRM} for k in names}
    st = sorted(starts) if starts else []
    j = 0
    for i in range(0, n_end, B):
        hit = False
        while j < len(st) and st[j] < i + B:
            hit = True
            j += 1
        if hit:
            for v in L.values():
                v.reset_per_prompt()
        x, y, idr, rec = xi[i:i + B], Y[i:i + B], ids[i:i + B], rec_all[i:i + B]
        with torch.no_grad():
            outs = {"reuse": x @ Wl.T, **{k: v.logits(x) for k, v in L.items()}}
            for k, lg in outs.items():
                mm = metrics(lg, idr, rec)
                for m in FLOORS + NRM:
                    acc[k][m].append(mm[m])
        for v in L.values():
            v.learn(x, y)
    res = {k: {m: torch.cat(acc[k][m]).numpy() for m in FLOORS + NRM} for k in names}
    return res, {name: L[name].n_params for name in L}


def prep(d, l):
    return f32(X[l - d]), f32(X[l]) @ W[l].T, torch.from_numpy(IDS[l].astype(np.int64))


# ---- 1. lr sweep on the first SWEEP_N tokens; criterion = mean over the four floors; global learners only
best = {}
for d in D_LIST:
    keys = {f"{name}@{lr:g}": (name, kind, lr, r, loss) for name, kind, lrs, r, loss in GRID for lr in lrs}
    sw = {k: [] for k in keys}
    for l in LAYERS:
        xi, Y, ids = prep(d, l)
        specs = {k: [(kind, lr, r, False, loss)] for k, (name, kind, lr, r, loss) in keys.items()}
        res, _ = simulate(xi, Y, ids, W[l], RECENT[l], specs, SWEEP_N)
        for k in keys:
            sw[k].append(np.mean([res[k][f].mean() for f in FLOORS]) * 100)
    print(f"\n=== lr sweep, d={d}, first {SWEEP_N} tokens: mean over layers {LAYERS} and floors {FLOORS} (%) ===")
    for name, kind, lrs, r, loss in GRID:
        vals = {lr: np.mean(sw[f"{name}@{lr:g}"]) for lr in lrs}
        for lr in lrs:
            print(f"   {name:14s} lr={lr:<8g} {vals[lr]:6.2f}")
        best[(d, name)] = max(lrs, key=lambda lr: vals[lr])
        print(f"   -> chosen {name} lr={best[(d, name)]:g}")
    log(f"sweep d={d} done")

# ---- 2. full stream: online-R, the cheap globals (listwise and MSE), and pp-G (per-prompt reset) for each cheap global
out, npar = {}, {}
for d in D_LIST:
    specs = {"online-R": [("full", best[(d, "online-R")], 0, False, "list")]}
    for name, kind, r in CHEAP:
        specs[name] = [(kind, best[(d, name)], r, False, "list")]
        specs[name + "/mse"] = [(kind, best[(d, name + "/mse")], r, False, "mse")]
        specs["pp-" + name] = [(kind, best[(d, name)], r, True, "list")]
    for l in LAYERS:
        xi, Y, ids = prep(d, l)
        res, npar[d] = simulate(xi, Y, ids, W[l], RECENT[l], specs, N, STARTS)
        out[(d, l)] = res
        log(f"d={d} layer {l} done")


def floor_mean(name):
    return np.mean([out[(2, l)][name][f].mean() for l in LAYERS for f in FLOORS])


# ---- 3. G+P: the best cheap listwise global (mean over floors, d=2) kept, plus a per-prompt bias or lowrank-8 part
best_cheap = max([n for n, _, _ in CHEAP], key=floor_mean)
g_kind, g_r = {n: (k, r) for n, k, r in CHEAP}[best_cheap]
COMBO = [("G+P(bias)", "bias", 0), ("G+P(lowrank-8)", "lowrank", 8)]
combo_out, combo_npar = {}, {}
if STARTS is not None:
    for d in D_LIST:
        specs = {}
        for cname, pkind, pr in COMBO:
            pname = "lowrank-8" if pkind == "lowrank" else "bias"
            specs[cname] = [(g_kind, best[(d, best_cheap)], g_r, False, "list"), (pkind, best[(d, pname)], pr, True, "list")]
        for l in LAYERS:
            xi, Y, ids = prep(d, l)
            res, combo_npar[d] = simulate(xi, Y, ids, W[l], RECENT[l], specs, N, STARTS)
            combo_out[(d, l)] = res
            log(f"G+P d={d} layer {l} done")

# ---- 4. report
STAR = lambda w: "*" if w * 1000 >= SHIFT_AT else " "
GLOBAL = ["online-R"] + [n for n, _, _ in CHEAP]
MSE = [n + "/mse" for n, _, _ in CHEAP]
PER = ["pp-" + n for n, _, _ in CHEAP]
COMBO_NAMES = [c for c, _, _ in COMBO] if STARTS is not None else []
NAMES = ["reuse"] + GLOBAL + MSE + PER + COMBO_NAMES


def get(d, l, n):
    return combo_out[(d, l)][n] if n.startswith("G+P") else out[(d, l)][n]


def lr_of(n, d):
    base = n.replace("pp-", "")
    return f"  [lr {best[(d, base)]:g}]" if (d, base) in best else ""


NR_LABEL = {"p1": "non-resident precision@1", "p2": "non-resident precision@2",
            "p4": "non-resident precision@4", "mr4": "non-resident miss-recall@4"}
for d in D_LIST:
    print(f"\n=== d={d}: lo2 = true ranks 0-1 in predicted top-4 (%), prequential, mean over layers {LAYERS}; * = after the shift ===")
    print("window(tokens)  " + " ".join(f"{n[:11]:>11}" for n in NAMES))
    for w in range(N // 1000):
        sl = slice(w * 1000, (w + 1) * 1000)
        row = [np.mean([get(d, l, n)["lo2"][sl].mean() for l in LAYERS]) * 100 for n in NAMES]
        print(f"{w*1000:6d}-{(w+1)*1000:<6d}{STAR(w)} " + " ".join(f"{v:11.1f}" for v in row))
    for metric in FLOORS + NRM:
        if metric in FLOORS:
            lo = LOS[FLOORS.index(metric)]
            lab = f"lo{lo}: true ranks 0-{lo-1} in predicted top-{2*lo}"
        else:
            lab = NR_LABEL[metric]
        print(f"--- d={d} {lab} (%): first 2000 | before shift | after shift   [lr]")

        def avg(n, a, b):
            return np.mean([np.nanmean(get(d, l, n)[metric][a:b]) for l in LAYERS]) * 100
        for n in NAMES:
            print(f"   {n:16s} {avg(n, 0, 2000):6.1f} | {avg(n, 2000, SHIFT_AT):6.1f} | {avg(n, SHIFT_AT, N):6.1f}{lr_of(n, d)}")
    if STARTS is not None:
        print(f"--- d={d} first 200 tokens of each prompt, lo2 and lo6 (%), over {len(STARTS)} prompts ({STARTS_WHY})")
        for n in NAMES:
            v2 = np.mean([get(d, l, n)["lo2"][s:s + 200].mean() for s in STARTS if s + 200 <= N for l in LAYERS]) * 100
            v6 = np.mean([get(d, l, n)["lo6"][s:s + 200].mean() for s in STARTS if s + 200 <= N for l in LAYERS]) * 100
            print(f"   {n:16s} {v2:6.1f} | {v6:6.1f}")
    else:
        print(f"--- d={d} (b) and (c) SKIPPED: prompt boundaries not found ({STARTS_WHY})")
    print(f"--- d={d} trainable parameters per layer (added to the router): " +
          ", ".join(f"{n} {npar[d][n]}" for n in GLOBAL + MSE + PER) +
          (", " + ", ".join(f"{n} {combo_npar[d][n]}" for n in COMBO_NAMES) if COMBO_NAMES else ""))
