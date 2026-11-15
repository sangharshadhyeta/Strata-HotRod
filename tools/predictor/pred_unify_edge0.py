"""Offline study for the unified expert predictor on an Edge0 router trace.
Target at layer l, lead 1: the model's own routing of layer l, from what is known one layer earlier.  Every learned model predicts the RESIDUAL of the
router logits (z_l - W_l x_{l-1}), i.e. it starts at router-reuse (what the engine's look-ahead does) and learns a correction by ridge - the safe start the
engine's learner has.  Features: H = x_{l-1} (the hidden input one layer earlier); C = one-hot of this token's layer l-1 picks; P = one-hot of the PREVIOUS
token's layer-l picks (an upper bound: inside a speculative window the previous row runs at the same time, so P is causal only for a window's first row).
Metric: recall of the TRUE rank 0-1 experts inside the predicted top-K (K = 2, 4, 8) and cov@10, on held-out steps (the last 30% of the windows)."""
import sys
import numpy as np
import torch
from trace_reader import read_trace, bf16_to_f32

torch.set_num_threads(8)
path = sys.argv[1]
max_rec = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "0" else None
lead = int(sys.argv[3]) if len(sys.argv) > 3 else 1
layer_step = int(sys.argv[4]) if len(sys.argv) > 4 else 4
d = read_trace(path, max_rec, 1)
order = np.lexsort((d["layer"], d["pos"], d["step"]))
step, pos, layer = d["step"][order], d["pos"][order], d["layer"][order]
x, ids = d["x"][order], d["ids"][order]
key = step.astype(np.uint64) * 16 + pos.astype(np.uint64)
uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
good = cnt == 48
first = first[good]
N = len(first)
rows = first[:, None] + np.arange(48)[None, :]
X = x[rows]
IDS = ids[rows].astype(np.int64)
STEP, POS = step[first], pos[first]
del x, ids
print("complete tokens", N, "windows", len(np.unique(STEP)), flush=True)

# previous row of the same window (pos >= 1): row index of (step, pos - 1) in the arrays above
idx_of = {(int(s), int(p)): i for i, (s, p) in enumerate(zip(STEP, POS))}
prev = np.array([idx_of.get((int(s), int(p) - 1), -1) for s, p in zip(STEP, POS)])
has_prev = prev >= 0

W = torch.from_numpy(np.load("/mnt/nvme/strata-data/traces/routers_edge0_f32.npy"))   # [48, 512, 2560]
ws = np.unique(STEP)
cut = ws[int(len(ws) * 0.7)]
tr = np.where(STEP < cut)[0]
te = np.where(STEP >= cut)[0]
print("train tokens", len(tr), "test tokens", len(te), "test rows with a previous row", int(has_prev[te].sum()), flush=True)


def onehot(a):          # a [n, 10] -> [n, 512] float32
    o = np.zeros((a.shape[0], 512), np.float32)
    np.put_along_axis(o, a, 1.0, axis=1)
    return o


def recall(logits, true_ids, ks=(2, 4, 8)):
    top = torch.topk(logits, 10, dim=1).indices.numpy()
    out = {}
    for K in ks:
        pk = top[:, :K]
        out[K] = float(np.mean([[e in p for e in r[:2]] for r, p in zip(true_ids, pk)]))
    out["cov10"] = float(np.mean([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(top, true_ids)]))
    return out


def ridge(F_tr, R_tr, lam):
    A = F_tr.T @ F_tr
    A += lam * torch.eye(A.shape[0]) * (torch.trace(A) / A.shape[0])
    return torch.linalg.solve(A, F_tr.T @ R_tr)


models = ["router-reuse", "ridge H", "ridge H+C", "ridge H+C+P (oracle)"]
acc = {m: [] for m in models}
lr_acc = {}
acc_base_p = []     # router-reuse and H+C on the rows that have a previous row (the matched comparison for the oracle)
acc_hc_p = []
for l in range(lead, 48, layer_step):
    xin = torch.from_numpy(bf16_to_f32(X[:, l - lead]))                  # [N, 2560]
    xl = torch.from_numpy(bf16_to_f32(X[:, l]))
    base = xin @ W[l].T                                                   # router-reuse logits [N, 512]
    z = xl @ W[l].T                                                       # the true router logits
    R = z - base
    mu, sd = xin[tr].mean(0), xin[tr].std(0) + 1e-6
    H = (xin - mu) / sd
    C = torch.from_numpy(onehot(IDS[:, l - lead]))
    P = torch.from_numpy(onehot(np.where(has_prev[:, None], IDS[np.maximum(prev, 0), l], 0)))
    P = P * torch.from_numpy(has_prev.astype(np.float32))[:, None]
    ones = torch.ones(N, 1)
    FH = torch.cat([H, ones], 1)
    FHC = torch.cat([H, C, ones], 1)
    FHCP = torch.cat([H, C, P, ones], 1)
    true_te = IDS[te, l]
    res = {}
    res["router-reuse"] = recall(base[te], true_te)
    for name, F, lam in (("ridge H", FH, 1.0), ("ridge H+C", FHC, 1.0), ("ridge H+C+P (oracle)", FHCP, 1.0)):
        Bm = ridge(F[tr], R[tr], lam)
        res[name] = recall(base[te] + F[te] @ Bm, true_te)
        if name == "ridge H+C":
            hp = np.where(has_prev[te])[0]
            acc_hc_p.append(recall((base[te] + F[te] @ Bm)[hp], true_te[hp]))
            acc_base_p.append(recall(base[te][hp], true_te[hp]))
        if name == "ridge H+C+P (oracle)":
            hp = np.where(has_prev[te])[0]
            res["oracle on rows with a previous row"] = recall((base[te] + F[te] @ Bm)[hp], true_te[hp])
    # the same correction at low rank (truncated SVD of the ridge-H solution): what a small learned correction keeps
    BH = ridge(FH[tr], R[tr], 1.0)
    U_, S_, Vh_ = torch.linalg.svd(BH[:-1], full_matrices=False)
    for r in (4, 8, 16, 32, 64):
        Br = (U_[:, :r] * S_[:r]) @ Vh_[:r]
        lr_acc.setdefault(r, []).append(recall(base[te] + FH[te][:, :-1] @ Br, true_te))
    for m in models:
        acc[m].append(res[m])
    acc.setdefault("oracle P rows", []).append(res["oracle on rows with a previous row"])
    print(f"layer {l}: " + " | ".join(f"{m} K4 {res[m][4]*100:.1f}" for m in models), flush=True)


def mean(lst, k):
    return np.mean([r[k] for r in lst]) * 100


print(f"\nlead {lead}, layers {lead}..47 step {layer_step}: true rank 0-1 in the predicted top-K, held-out windows")
print(f"{'model':30s} {'K=2':>6} {'K=4':>6} {'K=8':>6} {'cov@10':>7}")
for m in models:
    print(f"{m:30s} {mean(acc[m], 2):6.1f} {mean(acc[m], 4):6.1f} {mean(acc[m], 8):6.1f} {mean(acc[m], 'cov10'):7.1f}")
print("\nrows that have a previous row in their window (the oracle's rows):")
for m, lst in (("router-reuse", acc_base_p), ("ridge H+C", acc_hc_p), ("ridge H+C+P (oracle)", acc["oracle P rows"])):
    print(f"{m:30s} {mean(lst, 2):6.1f} {mean(lst, 4):6.1f} {mean(lst, 8):6.1f} {mean(lst, 'cov10'):7.1f}")

print("\nridge H at low rank (K=2 / K=4 / K=8 / cov@10):")
for r in sorted(lr_acc):
    print(f"rank {r:3d}: {mean(lr_acc[r], 2):5.1f} {mean(lr_acc[r], 4):5.1f} {mean(lr_acc[r], 8):5.1f} {mean(lr_acc[r], 'cov10'):5.1f}")
print(f"full    : {mean(acc['ridge H'], 2):5.1f} {mean(acc['ridge H'], 4):5.1f} {mean(acc['ridge H'], 8):5.1f} {mean(acc['ridge H'], 'cov10'):5.1f}")
