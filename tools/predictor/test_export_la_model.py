"""Tests for pt_reader.py and export_la_model.py: the torch-free reader against torch's own files, the published tar's layout,
the engine file's layout and the fold, and that a file with another global is refused.  Needs torch only to WRITE the test files
(skipped without it).

usage: python test_export_la_model.py
"""
import io
import os
import pickle
import struct
import tarfile
import tempfile
import zipfile

import numpy as np

import export_la_model as ex
import pt_reader

try:
    import torch
except ImportError:
    torch = None

H, E = 64, 512


def state(seed, extra_rows=8):
    """A layer's state as the published files hold it: out.weight [512 + extra, H] (the first 512 are the routing logits),
    out.bias, mu_state [1, H], sd_state [1, H]."""
    g = torch.Generator().manual_seed(seed)
    from collections import OrderedDict
    return OrderedDict([("out.weight", torch.randn(E + extra_rows, H, generator=g)), ("out.bias", torch.randn(E + extra_rows, generator=g)),
                        ("mu_state", torch.randn(1, H, generator=g)), ("sd_state", torch.rand(1, H, generator=g) + 0.5)])


def test_reader_matches_torch():
    with tempfile.TemporaryDirectory() as d:
        sd = state(1)
        sd["half"] = torch.randn(3, 5).half()
        sd["bf16"] = torch.randn(4).bfloat16()
        sd["strided"] = torch.randn(6, 7).t()                     # a non-contiguous view
        sd["scalar"] = torch.tensor(2.5)
        p = os.path.join(d, "s.pt")
        torch.save(sd, p)
        got = pt_reader.load(p)
        assert list(got) == list(sd)
        for k, v in sd.items():
            ref = v.float().numpy() if v.dtype == torch.bfloat16 else v.numpy()
            assert got[k].shape == ref.shape and np.array_equal(got[k], ref), k
        with open(p, "rb") as f:                                   # a file object, as the tar gives
            assert np.array_equal(pt_reader.load(f)["out.bias"], sd["out.bias"].numpy())


def test_reader_refuses_other_globals():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "evil.pt")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("archive/data.pkl", pickle.dumps(os.system))   # a global outside the allow-list
        try:
            pt_reader.load(p)
            raise AssertionError("a file with os.system was read")
        except pickle.UnpicklingError:
            pass


def tensor_pt(path, size, stride, offset=0, numel=4):
    """A torch.save zip by hand (no torch): one float tensor of `size` / `stride` / `offset` over a storage of `numel` floats."""
    s = lambda t: b"X" + struct.pack("<I", len(t)) + t.encode()          # BINUNICODE
    i = lambda v: b"J" + struct.pack("<i", v)                              # BININT
    tup = lambda vals: b"(" + b"".join(i(v) for v in vals) + b"t"          # MARK ... TUPLE
    pkl = (b"\x80\x02" + b"ctorch._utils\n_rebuild_tensor_v2\n" + b"("
           + b"(" + s("storage") + b"ctorch\nFloatStorage\n" + s("0") + s("cpu") + i(numel) + b"tQ"   # BINPERSID
           + i(offset) + tup(size) + tup(stride) + b"\x89" + b"tR.")      # requires_grad False, REDUCE, STOP
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pkl)
        z.writestr("archive/data/0", np.arange(numel, dtype=np.float32).tobytes())


def test_reader_refuses_a_tensor_past_its_storage():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ok.pt")
        tensor_pt(p, (2, 2), (2, 1))
        assert np.array_equal(pt_reader.load(p), np.arange(4, dtype=np.float32).reshape(2, 2))
        for size, stride, offset in (((100,), (1,), 0), ((2, 2), (2, 1), 1), ((4,), (-1,), 0)):
            tensor_pt(p, size, stride, offset)
            try:
                pt_reader.load(p)
                raise AssertionError(f"a tensor {size} / {stride} past its storage was read")
            except pickle.UnpicklingError:
                pass


def check_out(path, states, layers):
    with open(path, "rb") as f:
        assert f.read(4) == b"LAM1" and struct.unpack("<III", f.read(12)) == (layers, E, H)
        for l in range(layers):
            present = f.read(1)[0]
            assert present == (1 if l in states else 0), l
            if not present:
                continue
            w = (np.frombuffer(f.read(E * H * 2), np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(E, H)
            b = np.frombuffer(f.read(E * 4), np.float32)
            sd = {k: v.double().numpy() for k, v in states[l].items()}
            W, mu, s = sd["out.weight"][:E], sd["mu_state"].reshape(-1), sd["sd_state"].reshape(-1)
            assert np.allclose(w, W / s, rtol=1e-2, atol=1e-2)                       # bf16 of the folded matrix
            assert np.allclose(b, sd["out.bias"][:E] - (W / s) @ mu, rtol=1e-5, atol=1e-4)
        assert f.read() == b""


def test_export_from_folder_and_tar():
    layers = 6
    states = {l: state(10 + l) for l in (1, 2, 4, 5)}            # layer 3 missing: the router stays its look-ahead there
    with tempfile.TemporaryDirectory() as d:
        folder = os.path.join(d, "models_linear")
        os.makedirs(folder)
        torch.save(state(99, extra_rows=600), os.path.join(folder, "L00_d1_linear_state.pt"))   # layer 0: never written
        for l, sd in states.items():
            torch.save(sd, os.path.join(folder, f"L{l:02d}_d1_linear_state.pt"))
        with open(os.path.join(folder, "L01_d1_linear_state.json"), "w") as f:
            f.write("{}")                                           # the tar's .json companions are ignored
        out1 = os.path.join(d, "a.bin")
        ex.main([folder, out1, "--layers", str(layers), "--hidden", str(H)])
        check_out(out1, states, layers)
        tar = os.path.join(d, "197_models_d48_linear.tar")
        with tarfile.open(tar, "w") as t:                           # flat, as the published tar is
            for fn in sorted(os.listdir(folder)):
                t.add(os.path.join(folder, fn), arcname=fn)
        out2 = os.path.join(d, "b.bin")
        ex.main([tar, out2, "--layers", str(layers), "--hidden", str(H)])
        assert open(out1, "rb").read() == open(out2, "rb").read()
        empty = os.path.join(d, "empty")
        os.makedirs(empty)
        try:
            ex.main([empty, os.path.join(d, "c.bin"), "--layers", str(layers), "--hidden", str(H)])
            raise AssertionError("an empty folder made a file")
        except SystemExit:
            assert not os.path.exists(os.path.join(d, "c.bin")) and not os.path.exists(os.path.join(d, "c.bin.tmp"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if torch is None and name not in ("test_reader_refuses_other_globals", "test_reader_refuses_a_tensor_past_its_storage"):
                print("skipped (no torch)", name)
                continue
            fn()
            print("ok", name)
