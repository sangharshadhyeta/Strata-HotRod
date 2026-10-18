"""Extract every layer's router (ffn_gate_inp.weight, [n_embd=2560 -> 512]) from the Qwen GGUF into one float32 file for the analysis: routers_f32.npy [48, 512, 2560]."""
import sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

P = "/mnt/nvme/models/qwen38-flash-next/GSQ-RCO-IQ3_S/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
r = GGUFReader(P, "r")
by = {t.name: t for t in r.tensors}
W = []
for l in range(48):
    t = by[f"blk.{l}.ffn_gate_inp.weight"]
    if l == 0:
        print("router tensor:", list(t.shape), t.tensor_type.name)
    a = np.asarray(t.data)
    if t.tensor_type == GGMLQuantizationType.BF16:
        a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    a = a.astype(np.float32).reshape(512, 2560)          # gguf shape [2560, 512] = 512 rows of 2560
    W.append(a)
W = np.stack(W)
np.save("/mnt/nvme/strata-data/traces/routers_f32.npy", W)
print("saved", W.shape, W.dtype, "| |W| mean %.4f" % np.abs(W).mean())
