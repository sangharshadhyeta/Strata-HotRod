"""Writes the published per-layer d=1 linear expert predictors (aswinkumar99/qwen3.8-flash-next-expert-traces, models_linear/) as the
engine's STRATA_LA_MODEL file.  Their model for layer l on the previous layer's router input x:  logits = ((x - mu) / sd) @ W.T + b
(the first 512 outputs).  The standardisation is folded in:  W' = W / sd,  b' = b - W' @ mu,  so the engine runs the same bf16 GEMV
as the router, plus a bias.

File (little endian): u32 "LAM1", u32 n_layers, u32 512, u32 hidden; then per layer 0..n_layers-1: u8 present, and when present
bf16 W'[512][hidden], f32 b'[512].  (Layer 0 has no predictor: nothing comes before it.)

usage: python export_la_model.py <models_linear dir | 197_models_d48_linear.tar> <out.bin> [--layers 48] [--hidden 2560]
The published files: aswinkumar99/qwen3.8-flash-next-expert-traces, compact/heads/197/197_models_d48_linear.tar (L00..L47_d1_linear_state.pt;
setup --expert-predictor on downloads it).  The .pt files are read without torch (pt_reader.py: the same allow-list as torch.load's
weights_only - OrderedDict, the typed storages, _rebuild_tensor_v2 - anything else is refused).
The check at the end: on inputs drawn from each layer's own mu/sd, how often the folded bf16 model keeps the float model's top 2 / top 4."""
import argparse
import os
import struct
import sys

import tarfile

import numpy as np

from pt_reader import load as load_pt

E = 512


def bf16(a):
    """float32 -> bf16 bits, round to nearest even (as the engine's weights are stored)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def open_models(src):
    """-> a function layer -> its state dict, or None when `src` has no predictor for that layer (a folder or the published tar)."""
    if os.path.isdir(src):
        def get(l):
            path = os.path.join(src, f"L{l:02d}_d1_linear_state.pt")
            return load_pt(path) if os.path.exists(path) else None
        return get
    tf = tarfile.open(src)
    members = {os.path.basename(m.name): m for m in tf.getmembers() if m.isfile()}

    def get(l):
        m = members.get(f"L{l:02d}_d1_linear_state.pt")
        return load_pt(tf.extractfile(m)) if m is not None else None
    return get


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=2560)
    a = ap.parse_args(argv)
    get = open_models(a.models)
    rng = np.random.default_rng(0)
    agree2, agree4, n_written = [], [], 0
    with open(a.out + ".tmp", "wb") as f:
        f.write(b"LAM1" + struct.pack("<III", a.layers, E, a.hidden))
        for l in range(a.layers):
            sd = get(l) if l > 0 else None
            if sd is None:
                f.write(b"\x00")
                if l > 0:
                    print(f"layer {l}: no L{l:02d}_d1_linear_state.pt - the router stays its look-ahead", file=sys.stderr)
                continue
            W = np.asarray(sd["out.weight"][:E], np.float64)
            b = np.asarray(sd["out.bias"][:E], np.float64)
            mu = np.asarray(sd["mu_state"], np.float64).reshape(-1)
            s = np.asarray(sd["sd_state"], np.float64).reshape(-1)
            if W.shape != (E, a.hidden) or mu.shape != (a.hidden,) or s.shape != (a.hidden,):
                sys.exit(f"layer {l}: unexpected shapes W {W.shape} mu {mu.shape} sd {s.shape}")
            Wf = W / s[None, :]
            bfold = b - Wf @ mu
            wb = bf16(Wf.astype(np.float32))
            f.write(b"\x01")
            f.write(wb.tobytes())
            f.write(bfold.astype(np.float32).tobytes())
            n_written += 1
            # the fold's fidelity: float model vs folded bf16, on 512 inputs from this layer's own statistics
            x = mu[None, :] + s[None, :] * rng.standard_normal((512, a.hidden))
            ref = ((x - mu) / s) @ W.T + b
            got = x.astype(np.float32) @ bf16_to_f32(wb).astype(np.float32).T + bfold.astype(np.float32)
            for K, acc in ((2, agree2), (4, agree4)):
                r = np.argsort(-ref, axis=1)[:, :K]
                g = np.argsort(-got, axis=1)[:, :K]
                acc.append(np.mean([len(set(p) & set(q)) / K for p, q in zip(r.tolist(), g.tolist())]))
    if n_written == 0:
        os.remove(a.out + ".tmp")
        sys.exit(f"{a.models}: no L01..L{a.layers - 1:02d}_d1_linear_state.pt found")
    os.replace(a.out + ".tmp", a.out)
    print(f"{a.out}: {n_written} layers, {os.path.getsize(a.out) / 2**20:.0f} MiB; folded bf16 vs float: top-2 kept "
          f"{np.mean(agree2) * 100:.2f}%, top-4 kept {np.mean(agree4) * 100:.2f}% (worst layer {min(agree2) * 100:.2f}%)")


if __name__ == "__main__":
    main()
