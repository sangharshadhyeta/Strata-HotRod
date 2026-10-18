"""Transfer test: the published per-layer LINEAR predictors (aswinkumar99/qwen3.8-flash-next-expert-traces, d=1, trained on the NVFP4 checkpoint's activations)
run on OUR engine's router inputs (IQ3_S).  Arms at d = 1 on both test splits (IN, SHIFT):
   reuse     layer l's real router on x_{l-1}
   theirs    their model L{l}: z = (x_{l-1} - mu)/sd ; logits = z @ out.weight[:512].T + out.bias[:512]
   theirs+ft their weights fine-tuned on OUR training split (listwise KD against our own router's softmax, Adam, 3 epochs)
Loading is weights_only=True (the pickles only reference OrderedDict/FloatStorage/_rebuild_tensor_v2: checked)."""
import struct, sys, time
import numpy as np
import torch

torch.set_num_threads(10)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace1.bin"
MODELS = "/mnt/nvme/ref/trace-dataset/models_linear"
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
W = torch.from_numpy(np.load("/mnt/nvme/strata-data/traces/routers_f32.npy"))
f32 = lambda u: torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))
log("loaded", N, "tokens")

blk = (STEP_OF - STEP_OF.min()) // 40
te_in = (blk % 5 == 4)
te_shift = np.arange(N) >= int(0.77 * N)
splits = {"IN": (np.where(~te_in)[0], np.where(te_in)[0]), "SHIFT": (np.where(~te_shift)[0], np.where(te_shift)[0])}


def metrics(score, ids):
    top = score.topk(10, dim=1).indices.numpy()
    ids = ids.astype(np.int64)
    out = {f"r01@{K}": float((ids[:, :2, None] == top[:, None, :K]).any(2).mean()) for K in (2, 4, 8)}
    out["r03@8"] = float((ids[:, :4, None] == top[:, None, :8]).any(2).mean())
    out["cov10"] = float(np.mean([(len(set(a) & set(b)) / 10.0) for a, b in zip(top.tolist(), ids.tolist())]))
    return out


def theirs(l, xin):
    sd = torch.load(f"{MODELS}/L{l:02d}_d1_linear_state.pt", weights_only=True, map_location="cpu")
    mu, s = sd["mu_state"], sd["sd_state"]
    return sd["out.weight"][:E].clone(), sd["out.bias"][:E].clone(), mu, s


KEYS = ("r01@2", "r01@4", "r01@8", "r03@8", "cov10")
for split, (tr, te) in splits.items():
    agg = {"reuse": [], "theirs": [], "theirs+ft": []}
    for l in range(1, 48):
        Wo, bo, mu, sd = theirs(l, None)
        Xte = f32(X[te, l - 1]).reshape(len(te), H)
        ids_te = IDS[te, l]
        agg["reuse"].append(metrics(Xte @ W[l].T, ids_te))
        agg["theirs"].append(metrics(((Xte - mu) / sd) @ Wo.T + bo, ids_te))
        # fine-tune their weights on our training split: listwise distillation from OUR router
        Wf, bf = Wo.clone().requires_grad_(True), bo.clone().requires_grad_(True)
        opt = torch.optim.Adam([Wf, bf], lr=2e-4)
        Xtr = f32(X[tr, l - 1]).reshape(len(tr), H)
        Ztr = (Xtr - mu) / sd
        Ytr = torch.softmax(f32(X[tr, l]).reshape(len(tr), H) @ W[l].T, dim=1)
        for ep in range(3):
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 512):
                b = perm[i:i + 512]
                loss = -(Ytr[b] * torch.log_softmax(Ztr[b] @ Wf.T + bf, dim=1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            agg["theirs+ft"].append(metrics(((Xte - mu) / sd) @ Wf.T + bf, ids_te))
        if l % 8 == 0:
            log(split, "layer", l, "done")
    print(f"\n{split}: d=1, mean over layers 1..47")
    print(f"   {'':12s} " + " ".join(f"{k:>7}" for k in KEYS))
    for name, rows in agg.items():
        print(f"   {name:12s} " + " ".join(f"{np.mean([r[k] for r in rows]) * 100:7.1f}" for k in KEYS), flush=True)
