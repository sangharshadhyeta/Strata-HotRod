"""Weight-decay arm: the best cheap learner (lowrank-8/mse, stage-A settings) with AdamW decay on the correction parameters only.
Trace: trace_tok.bin (as stage A).  Layers 8,16,24,32,40; d=1 and d=2; full stream; lo3 (true ranks 0-2 in predicted top-6).
Decoupled decay: each step multiplies the correction by (1 - lr * wd) before the Adam step.
Output: tools/predictor/online_decay.out
"""
import struct, time
import numpy as np
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
H, NL, E = 2560, 48, 512
TRACE = "/mnt/nvme/strata-data/traces/trace_tok.bin"
ROUTER = "/mnt/nvme/strata-data/traces/routers_f32.npy"
OUT = "/home/Projects/Strata-train/tools/predictor/online_decay.out"
LAYERS5 = [8, 16, 24, 32, 40]
NEED = sorted({l - d for l in LAYERS5 for d in (1, 2)} | set(LAYERS5))
SETTINGS = {1: (1e-3, (0.1, 16)), 2: (1e-3, (0.1, 64))}          # stage A: lowrank-8/mse, chosen lr and margin/K per d
WDS = [0.0, 1e-4, 1e-3, 1e-2]
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
X = {l: np.zeros((N, H), np.uint16) for l in NEED}
IDS = {l: np.zeros((N, 10), np.int16) for l in LAYERS5}
tok = 0
for s in complete:
    for layer, nt, o, k in steps[s]:
        if layer not in X:
            continue
        region = np.asarray(buf[o + 8:o + 8 + nt * (H * 2 + k * 2)]).reshape(nt, H * 2 + k * 2)
        X[layer][tok:tok + nt] = np.ascontiguousarray(region[:, :H * 2]).view(np.uint16)
        if layer in IDS:
            IDS[layer][tok:tok + nt] = np.ascontiguousarray(region[:, H * 2:H * 2 + k * 2]).view(np.int16)[:, :10]
    tok += steps[s][0][1]
W = torch.from_numpy(np.load(ROUTER))
f32 = lambda u: torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))
SHIFT_AT = int(0.77 * N)
log("loaded", N, "tokens")


def metrics_lo3(logit, ids):
    top = logit.topk(12, dim=1).indices
    ids = ids.long()
    return (ids[:, :3, None] == top[:, None, :6]).any(2).float().mean(1)


def mse_loss(pred, Y, margin, K):
    v, i = Y.topk(K, dim=1)
    thr = v[:, -1:] - margin
    pc = pred.gather(1, i)
    dev = (pc - pc.mean(1, keepdim=True)) - (v - v.mean(1, keepdim=True))
    mask = torch.ones_like(pred).scatter_(1, i, 0.0)
    hinge = (torch.relu(pred - thr) * mask).sum(1) / (E - K)
    return (dev ** 2).mean() + hinge.mean()


def run(l, d, wd):
    lr, cfg = SETTINGS[d]
    xi = f32(X[l - d])
    Y = f32(X[l]) @ W[l].T
    ids = torch.from_numpy(IDS[l].astype(np.int64))
    A = (torch.randn(8, H) * 1e-3).requires_grad_(True)
    Bm = torch.zeros(E, 8, requires_grad=True)
    opt = torch.optim.AdamW([A, Bm], lr=lr, weight_decay=wd)
    base_router = W[l]
    out = []
    for i in range(0, N, B):
        x, y, idr = xi[i:i + B], Y[i:i + B], ids[i:i + B]
        with torch.no_grad():
            base = x @ base_router.T
            out.append(metrics_lo3(base + (x @ A.T) @ Bm.T, idr))
        logit = x @ base_router.T + (x @ A.T) @ Bm.T
        loss = mse_loss(logit, y, cfg[0], cfg[1])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.cat(out).numpy()


with open(OUT, "w") as F:
    def emit(*a):
        print(*a, file=F, flush=True)
    emit("DECAY ARM: lowrank-8/mse (stage-A settings), AdamW weight decay on the correction only, trace_tok.bin, layers "
         f"{LAYERS5}, lo3 = true ranks 0-2 in predicted top-6 (%)")
    for d in (1, 2):
        lr, cfg = SETTINGS[d]
        emit(f"\n--- d={d}, lr={lr:g}, margin={cfg[0]} K={cfg[1]}: first 2000 | before shift | after shift (mean over layers)")
        reuse = {}
        dec = {wd: {} for wd in WDS}
        for l in LAYERS5:
            xi = f32(X[l - d])
            ids = torch.from_numpy(IDS[l].astype(np.int64))
            with torch.no_grad():
                reuse[l] = metrics_lo3(xi @ W[l].T, ids).numpy()
            for wd in WDS:
                dec[wd][l] = run(l, d, wd)
            log(f"d={d} layer {l} done")

        def row(arr):
            return " | ".join(f"{np.mean([np.nanmean(arr[l][a:b]) for l in LAYERS5]) * 100:6.1f}" for a, b in ((0, 2000), (2000, SHIFT_AT), (SHIFT_AT, N)))
        emit(f"   reuse (router)       {row(reuse)}")
        for wd in WDS:
            emit(f"   wd={wd:<8g}         {row(dec[wd])}")
    log("written", OUT)
