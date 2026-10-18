"""Live-learning simulation (prequential: every token is SCORED with the current weights first, then LEARNED from).
The trace is streamed in its original order: code -> agent -> reasoning -> prose -> other languages -> summaries, so the shift at ~77% is a real
topic change.  Learners (per layer l, lead d; input = router input of layer l-d, target = layer l's true routing):
  reuse     layer l's real router on x_{l-d}; never learns (the baseline Strata has today)
  online-R  weights start AT the real router (so it starts at the baseline), Adam on a listwise loss against the true routing weights
  online-Z  the same learner started from ZERO weights (solid's situation)
  solid     solid.cpp's rule: logistic regression, zero init, the 10 chosen experts are positives and 8 random others negatives, plain SGD
  prior+R   ridge trained OFFLINE on the first 60% of the stream, then online-R-style Adam from there (an offline prior that keeps adapting)
Engine realism: the label is the layer's routed top-10 with its routing weights (softmax over the 10 chosen logits), which the engine has;
updates in mini-batches of 16 tokens (labels a few tokens late).  Score: true rank 0-1 inside the predicted top-4, and top-10 coverage."""
import struct, sys, time
import numpy as np
import torch

torch.set_num_threads(10)
torch.manual_seed(0)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace1.bin"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

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
log("loaded", N, "tokens, layers", need)

B, WIN = 16, 1000
PRIOR_END = int(0.60 * N)
SHIFT_AT = int(0.77 * N)


def score(logit, ids):
    top = logit.topk(10, dim=1).indices
    ids = ids.long()
    r4 = (ids[:, :2, None] == top[:, None, :4]).any(2).float().mean(1)          # per-token: share of rank 0-1 inside top-4
    cov = torch.tensor([len(set(a) & set(b)) / 10.0 for a, b in zip(top.tolist(), ids.tolist())])
    return r4, cov


def target(Y):                                                                    # routing weights: softmax over the chosen top-10 logits
    v, i = Y.topk(10, dim=1)
    p = torch.zeros_like(Y)
    p.scatter_(1, i, torch.softmax(v, dim=1))
    return p


class Learner:
    def __init__(self, name, Winit, lr, kind="kl"):
        self.name, self.kind = name, kind
        self.W = Winit.clone().requires_grad_(kind == "kl")
        self.b = torch.zeros(E, requires_grad=(kind == "kl"))
        self.lr = lr
        self.opt = torch.optim.Adam([self.W, self.b], lr=lr) if kind == "kl" else None

    def logits(self, x):
        return x @ self.W.T + self.b

    def learn(self, x, Y, ids):
        if self.kind == "kl":
            loss = -(target(Y) * torch.log_softmax(self.logits(x), dim=1)).sum(1).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        else:                                                                     # solid's rule, per sample, no autograd
            with torch.no_grad():
                for xi, idr in zip(x, ids):
                    pos = idr.long()
                    neg = torch.randint(0, E, (8,))
                    neg = neg[~torch.isin(neg, pos)]
                    for e, lab in [(int(e), 1.0) for e in pos] + [(int(e), 0.0) for e in neg]:
                        p = torch.sigmoid(self.W[e] @ xi)
                        self.W[e] -= self.lr * (p - lab) * xi


out = {}
for d in (1, 2):
    for l in LAYERS:
        xi_all = f32(X[l - d])
        Y_all = f32(X[l]) @ W[l].T
        ids_all = torch.from_numpy(IDS[l].astype(np.int64))
        # offline prior: ridge on the first 60%
        tr = slice(0, PRIOR_END)
        mu, ymu = xi_all[tr].mean(0, keepdim=True), Y_all[tr].mean(0, keepdim=True)
        G = (xi_all[tr] - mu).T @ (xi_all[tr] - mu) / PRIOR_END
        Cm = (xi_all[tr] - mu).T @ (Y_all[tr] - ymu) / PRIOR_END
        Wr = torch.linalg.solve(G + 0.3 * float(G.diagonal().mean()) * torch.eye(H), Cm).T              # [512, 2560]
        learners = {"online-R": Learner("online-R", W[l], 1e-4), "online-Z": Learner("online-Z", torch.zeros(E, H), 1e-3),
                    "solid": Learner("solid", torch.zeros(E, H), 1e-3, "solid")}
        prior = Learner("prior+R", Wr, 1e-4)
        prior.b = (ymu - mu @ Wr.T).squeeze(0).clone().requires_grad_(True)
        prior.opt = torch.optim.Adam([prior.W, prior.b], lr=1e-4)
        res = {k: [] for k in ("reuse", "online-R", "online-Z", "solid", "prior+R")}
        cov = {k: [] for k in res}
        for i in range(0, N, B):
            x, Y, ids = xi_all[i:i + B], Y_all[i:i + B], ids_all[i:i + B]
            with torch.no_grad():
                s_reuse = x @ W[l].T
                outs = {"reuse": s_reuse, **{k: v.logits(x) for k, v in learners.items()}, "prior+R": prior.logits(x) if i >= PRIOR_END else None}
            for k, lg in outs.items():
                if lg is None:
                    res[k].extend([float("nan")] * len(x)); cov[k].extend([float("nan")] * len(x))
                    continue
                r4, c = score(lg, ids)
                res[k].extend(r4.tolist()); cov[k].extend(c.tolist())
            for v in learners.values():
                v.learn(x, Y, ids)
            if i >= PRIOR_END:
                prior.learn(x, Y, ids)
        out[(d, l)] = (res, cov)
        log(f"d={d} layer {l} done")

# ---- report: per 1000-token window, mean over layers; the stream order is code -> agent -> reasoning -> prose -> other languages -> summaries
for d in (1, 2):
    print(f"\n=== d={d}: share of TRUE rank 0-1 experts inside the predicted top-4 (prequential), mean over layers {LAYERS}; * = after the shift ===")
    names = ("reuse", "online-R", "online-Z", "solid", "prior+R")
    print("window(tokens)   " + " ".join(f"{n:>9}" for n in names))
    nwin = N // WIN
    for w in range(nwin):
        sl = slice(w * WIN, (w + 1) * WIN)
        row = []
        for n in names:
            v = np.nanmean([np.nanmean(out[(d, l)][0][n][sl]) for l in LAYERS]) * 100 if not all(np.isnan(out[(d, l)][0][n][sl]).all() for l in LAYERS) else float("nan")
            row.append(v)
        print(f"{w*WIN:6d}-{(w+1)*WIN:<6d}{'*' if w * WIN >= SHIFT_AT else ' '} " + " ".join(f"{v:9.1f}" for v in row))
    print("--- averages over the stream (first 2000 tokens = learning from scratch | before the shift | after the shift):")
    for n in names:
        def avg(a, b):
            vals = [np.nanmean(out[(d, l)][0][n][a:b]) for l in LAYERS if not np.isnan(out[(d, l)][0][n][a:b]).all()]
            return np.mean(vals) * 100 if vals else float("nan")
        print(f"   {n:9s} {avg(0, 2000):6.1f} | {avg(2000, SHIFT_AT):6.1f} | {avg(SHIFT_AT, N):6.1f}")
