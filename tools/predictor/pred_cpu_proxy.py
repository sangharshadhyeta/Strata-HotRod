"""How many of the CPU's rank 0-1 computations would a predictor avoid?  Proxy for the GPU residency: per layer, the 58 most frequent experts of the TRAINING part
(58 x 48 = 2,784 ~ the 2,793 experts the 3060 holds in production).  The CPU computes a true rank 0-1 expert only when it is NOT resident.  A predictor
admits K extra experts per layer (its top-K among the non-resident ones); the share of those CPU-bound rank 0-1 experts it covers = computations avoided.
Comparators: 'static +K' = just keep the K next most frequent experts per layer resident (no prediction at all).  Test = the last 23% (other languages,
summaries); the hot set and nothing else comes from the first 77%.  Their model (d=1) is used as published, never trained on our data."""
import struct, sys, time
import numpy as np
import torch

torch.set_num_threads(10)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace1.bin"
MODELS = "/mnt/nvme/ref/trace-dataset/models_linear"
buf = np.memmap(TRACE, dtype=np.uint8, mode="r")
off, recs = 4, []
while off + 8 <= len(buf):
    step, layer, nt, k = struct.unpack_from("<IHBB", buf, off)
    rec = 8 + nt * (H * 2 + k * 2)
    if off + rec > len(buf):
        break
    recs.append((step, layer, nt, off, k)); off += rec
steps = {}
for step, layer, nt, o, k in recs:
    steps.setdefault(step, []).append((layer, nt, o, k))
complete = sorted(s for s, v in steps.items() if len(v) == NL and len({nt for _, nt, _, _ in v}) == 1)
N = sum(steps[s][0][1] for s in complete)
X = np.zeros((N, NL, H), np.uint16); IDS = np.zeros((N, NL, 10), np.int16)
tok = 0
for s in complete:
    for layer, nt, o, k in steps[s]:
        p = o + 8
        for t in range(nt):
            X[tok + t, layer] = np.frombuffer(buf[p:p + H * 2], dtype=np.uint16); p += H * 2
            IDS[tok + t, layer] = np.frombuffer(buf[p:p + k * 2], dtype=np.int16)[:10]; p += k * 2
    tok += steps[s][0][1]
W = torch.from_numpy(np.load("/mnt/nvme/strata-data/traces/routers_f32.npy"))
f32 = lambda u: torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))
cut = int(0.77 * N)
tr, te = np.arange(cut), np.arange(cut, N)
RES = 58
KS = (1, 2, 4, 8)
agg = {name: {K: [0, 0] for K in KS} for name in ("static +K", "reuse d=1", "theirs d=1", "reuse d=2")}
base_total, base_cpu = 0, 0
for l in range(2, 48):
    cnt = np.bincount(IDS[tr, l].reshape(-1).astype(np.int64), minlength=E)
    order = np.argsort(-cnt)
    resident = np.zeros(E, bool); resident[order[:RES]] = True
    true01 = torch.from_numpy(IDS[te, l, :2].astype(np.int64))                  # [n,2]
    nonres = ~resident[true01.numpy()]                                           # [n,2] this rank 0-1 expert would go to the CPU
    base_total += nonres.size; base_cpu += int(nonres.sum())
    sd = torch.load(f"{MODELS}/L{l:02d}_d1_linear_state.pt", weights_only=True, map_location="cpu")
    scores = {}
    x1, x2 = f32(X[te, l - 1]), f32(X[te, l - 2])
    scores["reuse d=1"] = x1 @ W[l].T
    scores["theirs d=1"] = ((x1 - sd["mu_state"]) / sd["sd_state"]) @ sd["out.weight"][:E].T + sd["out.bias"][:E]
    scores["reuse d=2"] = x2 @ W[l].T
    scores["static +K"] = torch.from_numpy(cnt.astype(np.float32))[None, :].expand(len(te), E).clone()
    for name, sc in scores.items():
        sc = sc.clone()
        sc[:, torch.from_numpy(resident)] = -1e30                                # only experts NOT already resident can be admitted
        top = sc.topk(8, dim=1).indices
        for K in KS:
            hit = (true01[:, :, None] == top[:, None, :K]).any(2).numpy()        # [n,2]
            agg[name][K][0] += int((hit & nonres).sum()); agg[name][K][1] += int(nonres.sum())
print(f"hot set = {RES} most frequent experts per layer (train part); test = {len(te)} tokens of other languages / summaries")
print(f"true rank 0-1 experts that are NOT resident (would go to the CPU): {base_cpu / base_total * 100:.1f}% of all rank 0-1 picks")
print("share of those CPU computations AVOIDED when K extra experts per layer are admitted (K = predicted top-K among the non-resident):")
print(f"   {'':12s} " + " ".join(f"K={K:<5d}" for K in KS))
for name, v in agg.items():
    print(f"   {name:12s} " + " ".join(f"{v[K][0] / max(v[K][1], 1) * 100:6.1f}%" for K in KS))
