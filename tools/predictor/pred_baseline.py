"""Consistency check + router-reuse baseline on a (partial) router trace.
Consistency: recomputing W_l @ x_l must reproduce the recorded top-10 (the recorded input IS what the router reads, ids are in rank order).
Baseline (Fate / HOBBIT's stacked gate, what Strata's RouterLookahead already does): layer l's real router applied to the input of layer l-d, d = 1..4.
Metrics: recall of the TRUE rank 0-1 experts (the ones the CPU computes today) and of ranks 0-3, inside the predicted top-K, K = 2, 4, 8, 10;
         cov@10 = |predicted top 10 & true top 10| / 10 (the prior study's metric)."""
import sys
import numpy as np
import torch
from trace_reader import read_trace, bf16_to_f32

torch.set_num_threads(8)
path = sys.argv[1]
max_rec = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '0' else None
step_mod = int(sys.argv[3]) if len(sys.argv) > 3 else 1
d = read_trace(path, max_rec, step_mod)
print("records", d["records"], "token rows", len(d["layer"]), flush=True)

# group rows into [token, layer]: a token row is one (step, pos); keep only tokens that have all 48 layers
order = np.lexsort((d["layer"], d["pos"], d["step"]))
step, pos, layer = d["step"][order], d["pos"][order], d["layer"][order]
x, ids = d["x"][order], d["ids"][order]
key = step.astype(np.uint64) * 16 + pos.astype(np.uint64)
uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
good = cnt == 48
first = first[good]
N = len(first)
rows = first[:, None] + np.arange(48)[None, :]                    # [N, 48] indices into the sorted arrays
assert (layer[rows] == np.arange(48)[None, :]).all()
X = x[rows]                                                       # uint16 [N, 48, 2560]
IDS = ids[rows]                                                   # int16  [N, 48, 10]
print("complete tokens:", N, "| tokens/step (mean window):", round(N / len(np.unique(step[first])), 2), flush=True)
W = torch.from_numpy(np.load("/mnt/nvme/strata-data/traces/routers_f32.npy"))   # [48, 512, 2560]


def logits(l_router, l_input, rows_idx):
    xi = torch.from_numpy(bf16_to_f32(X[rows_idx, l_input]))       # [n, 2560]
    return xi @ W[l_router].T                                      # [n, 512]


sel = np.arange(0, N, max(1, N // 6000))                           # at most ~6000 tokens, spread over the whole trace
n = len(sel)
print("scored tokens:", n, flush=True)

# 1) consistency
agree1, agree10 = [], []
for l in range(0, 48, 6):
    top = logits(l, l, sel).topk(10, dim=1).indices.numpy()
    rec = IDS[sel, l].astype(np.int64)
    agree1.append((top[:, 0] == rec[:, 0]).mean())
    agree10.append(np.mean([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(top, rec)]))
print("consistency (layers 0,6,...,42): rank-0 agrees %.4f | top-10 set overlap %.4f" % (np.mean(agree1), np.mean(agree10)), flush=True)

# 2) router-reuse baseline
print("\nrouter-reuse baseline: layer l's router applied to the input of layer l-d")
print(f"{'d':>2} | {'rank0-1 in top-K: K=2':>22} {'K=4':>7} {'K=8':>7} {'K=10':>7} | {'ranks0-3 K=8':>12} | {'cov@10':>7}")
res = {}
for dd in (1, 2, 3, 4):
    hit = {K: [] for K in (2, 4, 8, 10)}
    r3 = []
    cov = []
    for l in range(dd, 48):
        lg = logits(l, l - dd, sel)
        top = lg.topk(10, dim=1).indices.numpy()
        rec = IDS[sel, l].astype(np.int64)
        for K in hit:
            pk = top[:, :K]
            hit[K].append(np.mean([[e in p for e in r[:2]] for r, p in zip(rec, pk)]))
        r3.append(np.mean([[e in p for e in r[:4]] for r, p in zip(rec, top[:, :8])]))
        cov.append(np.mean([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(top, rec)]))
    res[dd] = {K: float(np.mean(v)) for K, v in hit.items()}
    print(f"{dd:>2} | {np.mean(hit[2])*100:22.1f} {np.mean(hit[4])*100:7.1f} {np.mean(hit[8])*100:7.1f} {np.mean(hit[10])*100:7.1f} | {np.mean(r3)*100:12.1f} | {np.mean(cov)*100:7.1f}", flush=True)
np.save("/mnt/nvme/strata-data/traces/baseline_partial.npy", np.array([[res[dd][K] for K in (2, 4, 8, 10)] for dd in (1, 2, 3, 4)]))
