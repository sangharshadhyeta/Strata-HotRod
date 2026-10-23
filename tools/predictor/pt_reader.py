"""Reads a PyTorch .pt state file (the zip format torch.save writes) into numpy arrays without torch.

Only plain tensors in (Ordered)dicts are understood, the same allow-list torch.load(weights_only=True) checks for these files
(OrderedDict, the typed storages, _rebuild_tensor_v2); any other global in the pickle is refused, so a hostile file cannot run
code.  Used by export_la_model.py, so setup can turn the published expert predictors into the engine's file without torch.
"""
import collections
import io
import pickle
import zipfile

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64,
           "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8,
           "BoolStorage": np.bool_, "BFloat16Storage": "bf16"}


class _Storage:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *_):
    a = storage
    # as_strided does not check its bounds: a size, stride or offset past the storage would read memory outside it
    if not isinstance(a, np.ndarray) or not isinstance(offset, int) or offset < 0 or \
            len(size) != len(stride) or any(not isinstance(v, int) or v < 0 for v in (*size, *stride)):
        raise pickle.UnpicklingError("refused tensor record")
    if not size:
        if offset >= a.size:
            raise pickle.UnpicklingError("refused tensor record: past its storage")
        return a[offset].copy()
    if 0 not in size and offset + sum((n - 1) * s for n, s in zip(size, stride)) >= a.size:
        raise pickle.UnpicklingError("refused tensor record: past its storage")
    item = a.itemsize
    return np.lib.stride_tricks.as_strided(a[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)).copy()


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name in _DTYPES:
            return _Storage(name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        raise pickle.UnpicklingError(f"refused global {module}.{name}: not a plain tensor file")

    def persistent_load(self, pid):
        kind, stype, key, _loc, numel = pid[:5]
        if kind != "storage" or not isinstance(stype, _Storage):
            raise pickle.UnpicklingError("refused storage record")
        raw = self.zf.read(f"{self.prefix}data/{key}")
        dt = _DTYPES[stype.name]
        if dt == "bf16":
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)[:numel]
        return np.frombuffer(raw, dt)[:numel]


def load(src):
    """`src`: a path or a binary file object of a torch.save zip.  -> its dict, tensors as numpy arrays (bf16 as float32)."""
    with zipfile.ZipFile(src) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]
        return _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
