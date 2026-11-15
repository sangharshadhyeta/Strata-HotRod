"""Edge0's routers: every layer's ffn_gate_inp.weight [2560 -> 512] from model-skeleton.gguf into one float32 file (the framework's router_weights.py, for Edge0)."""
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
P = "/mnt/nvme/models/edge0-145b-nvfp4/model-skeleton.gguf"
r = GGUFReader(P, "r")
by = {t.name: t for t in r.tensors}
print("gate_inp tensors:", sorted(n for n in by if "gate_inp" in n)[:3], "...", sum(1 for n in by if "gate_inp" in n))
W = []
for l in range(48):
    t = by[f"blk.{l}.ffn_gate_inp.weight"]
    if l == 0:
        print("router tensor:", list(t.shape), t.tensor_type.name)
    a = np.asarray(t.data)
    if t.tensor_type == GGMLQuantizationType.BF16:
        a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    a = a.astype(np.float32).reshape(512, 2560)
    W.append(a)
W = np.stack(W)
np.save("/mnt/nvme/strata-data/traces/routers_edge0_f32.npy", W)
print("saved", W.shape, W.dtype, "| |W| mean %.4f" % np.abs(W).mean())
