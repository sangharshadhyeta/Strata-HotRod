"""Does the router following the pre-router beat the stand-in rule?  Offline, on the Edge0 trace, per layer l:
resident set R = the n_res most frequent experts of the training windows (what a frequency-filled card holds); the pre-router (ridge on the hidden state,
lead 1, trained on the training windows, starting at the real router) names its top-K for each test token and those K are made ready for that token.
Measured on held-out windows: hit = share of the final picks that are on the card; swapped = the share of the router's own weight on picks that were changed.
  stand-in rule: picks of rank >= lo that are not available are replaced by the best available expert (highest logit, not already picked) when
                 router weight x logit gap <= budget (the engine's rule), else stay exact (fetched: a miss).
  bias:          top-10 of (logit + beta * available) (the router follows what is on the card).
available = R, or R + the pre-router's top-K for the token (K = 0, 2, 4, 8)."""
import sys
import numpy as np
import torch
from trace_reader import read_trace, bf16_to_f32

torch.set_num_threads(8)
path = sys.argv[1]
n_res = int(sys.argv[2]) if len(sys.argv) > 2 else 25
LO = 3
LEAD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
d = read_trace(path, None, 1)
order = np.lexsort((d["layer"], d["pos"], d["step"]))
step, pos, layer = d["step"][order], d["pos"][order], d["layer"][order]
x, ids = d["x"][order], d["ids"][order]
key = step.astype(np.uint64) * 16 + pos.astype(np.uint64)
uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
first = first[cnt == 48]
N = len(first)
rows = first[:, None] + np.arange(48)[None, :]
X = x[rows]
IDS = ids[rows].astype(np.int64)
STEP = step[first]
del x, ids
W = torch.from_numpy(np.load("/mnt/nvme/strata-data/traces/routers_edge0_f32.npy"))
ws = np.unique(STEP)
cut = ws[int(len(ws) * 0.7)]
tr, te = np.where(STEP < cut)[0], np.where(STEP >= cut)[0]
print("tokens", N, "train", len(tr), "test", len(te), "resident per layer", n_res, flush=True)


def ridge(F, R, lam=1.0):
    A = F.T @ F
    A += lam * torch.eye(A.shape[0]) * (torch.trace(A) / A.shape[0])
    return torch.linalg.solve(A, F.T @ R)


def route(z, k=10):
    top = torch.topk(z, k, dim=1)
    w = torch.softmax(top.values, dim=1)                       # the router's weights: softmax over the ten chosen logits
    return top.indices, w


def standin(z, avail, budget):
    """z [n,512], avail [n,512] bool.  Returns hit, swapped, miss (picks neither available nor replaced)."""
    idx, w = route(z)
    n = z.shape[0]
    picked = torch.zeros_like(z, dtype=torch.bool)
    picked.scatter_(1, idx, True)
    cand = z.masked_fill(~avail | picked, -1e30)                # resident experts not already picked
    hits = swapped = 0.0
    for r in range(10):
        e = idx[:, r]
        have = avail.gather(1, e[:, None])[:, 0]
        ok = have.clone()
        sw = torch.zeros(n, dtype=torch.bool)
        if r >= LO:
            best = cand.max(dim=1)
            gap = (z.gather(1, e[:, None])[:, 0] - best.values).clamp(min=0)
            take = (~have) & (best.values > -1e29) & (w[:, r] * gap <= budget)
            sw = take
            ok = have | take
            cand.scatter_(1, best.indices[:, None], torch.where(take[:, None], torch.full((n, 1), -1e30), cand.gather(1, best.indices[:, None])))
        hits += ok.float().mean().item()
        swapped += (w[:, r] * sw.float()).mean().item()
    return hits / 10, swapped


def bias(z, avail, beta):
    idx, w = route(z)
    idx2 = torch.topk(z + beta * avail.float(), 10, dim=1).indices
    hit = avail.gather(1, idx2).float().mean().item()
    in2 = torch.zeros_like(z, dtype=torch.bool); in2.scatter_(1, idx2, True)
    dropped = ~in2.gather(1, idx)                              # the router's own picks that are no longer picked
    return hit, float((w * dropped.float()).sum(1).mean().item())


LAYERS = [8, 16, 24, 32, 40]
BUDGETS = [0.0, 0.03, 0.1, 0.3, 1.0, 1e9]
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0]
KS = [0, 2, 4, 8]
out = {}
for l in LAYERS:
    xin = torch.from_numpy(bf16_to_f32(X[:, l - LEAD]))
    z = torch.from_numpy(bf16_to_f32(X[:, l])) @ W[l].T
    base = xin @ W[l].T
    mu, sd = xin[tr].mean(0), xin[tr].std(0) + 1e-6
    F = torch.cat([(xin - mu) / sd, torch.ones(N, 1)], 1)
    Bm = ridge(F[tr], (z - base)[tr])
    pred = base + F @ Bm                                        # the pre-router (lead 1)
    freq = np.bincount(IDS[tr, l].ravel(), minlength=512)
    R = torch.zeros(512, dtype=torch.bool)
    R[torch.from_numpy(np.argsort(-freq)[:n_res].copy())] = True
    zt, bt, pt = z[te], base[te], pred[te]
    for name, pz in (("router-reuse", bt), ("pre-router", pt)):
        for K in KS:
            avail = R[None, :].expand(len(te), 512).clone()
            if K:
                avail.scatter_(1, torch.topk(pz, K, dim=1).indices, True)
            for b in BUDGETS:
                h, s = standin(zt, avail, b)
                out.setdefault(("standin", name, K, b), []).append((h, s))
            for beta in BETAS:
                h, s = bias(zt, avail, beta)
                out.setdefault(("bias", name, K, beta), []).append((h, s))
    print("layer", l, "done", flush=True)


def m(k):
    a = np.array(out[k])
    return a[:, 0].mean() * 100, a[:, 1].mean() * 100


print(f"\nresident {n_res}/512 per layer; hit = picks on the card (%), swapped = router weight on changed picks (%)")
for name in ("router-reuse", "pre-router"):
    for K in KS:
        if name == "router-reuse" and K == 0:
            pass
        elif K == 0 and name == "pre-router":
            continue
        print(f"-- ready for the token: R + top-{K} of {name}")
        print("   stand-in rule: " + "  ".join(f"b={b:g}: hit {m(('standin', name, K, b))[0]:.1f}/sw {m(('standin', name, K, b))[1]:.1f}" for b in BUDGETS))
        print("   router bias  : " + "  ".join(f"beta={be:g}: hit {m(('bias', name, K, be))[0]:.1f}/sw {m(('bias', name, K, be))[1]:.1f}" for be in BETAS))
