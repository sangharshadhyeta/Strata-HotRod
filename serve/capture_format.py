"""capture_format.py - the on-disk format of the capture ring (docs/SPEC-capture.md), shared by serve/capture.py (the writer)
and serve/capture_read.py (the reader).  No regex, no text: the records hold token ids and numbers.

A segment file is one header and then records, appended whole:

  header:  MAGIC (9 bytes) | u32 version | u32 json length | JSON {format, fingerprint, model, seq, created}
  record:  u32 body length | u32 crc32 of the body | body
  body:    u8 kind code (1=A 2=B 3=C 4=D 5=E) | u32 json length | JSON meta | the arrays, raw, in meta["arrays"] order

meta["arrays"] is a list of [name, dtype, count]; dtype is "i4" (int32 token ids), "f4" (float32) or "f2" (float16 raw bytes).
All numbers are little-endian.  A record whose length or crc does not check ends its segment (a torn tail after a crash).
"""
import array
import json
import struct
import sys
import zlib

MAGIC = b"STRATACAP"
VERSION = 1
KINDS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
CODE_KIND = {v: k for k, v in KINDS.items()}
DTYPE_SIZE = {"i4": 4, "f4": 4, "f2": 2}
_TYPECODE = {"i4": "i", "f4": "f"}
_HDR = struct.Struct("<II")
_REC = struct.Struct("<II")
_U32 = struct.Struct("<I")


def encode_header(fingerprint: str, model: str, seq: int, created: float) -> bytes:
    body = json.dumps({"format": VERSION, "fingerprint": fingerprint, "model": model, "seq": seq,
                       "created": created}, separators=(",", ":")).encode("utf-8")
    return MAGIC + _HDR.pack(VERSION, len(body)) + body


def read_header(f) -> dict:
    """The header at the start of an open segment file.  ValueError for anything that is not one of ours."""
    magic = f.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError("not a capture segment (bad magic)")
    fixed = f.read(_HDR.size)
    if len(fixed) != _HDR.size:
        raise ValueError("capture segment header is cut short")
    version, n = _HDR.unpack(fixed)
    if version != VERSION:
        raise ValueError(f"capture segment format {version} is not format {VERSION}")
    raw = f.read(n)
    if len(raw) != n:
        raise ValueError("capture segment header is cut short")
    try:
        header = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise ValueError(f"capture segment header is not JSON ({e})") from None
    if not isinstance(header, dict) or header.get("format") != VERSION:
        raise ValueError("capture segment header has no format")
    return header


def _le_bytes(dtype: str, data) -> bytes:
    """The raw little-endian bytes of one array: an array.array of the dtype's item size, or bytes already in that layout."""
    if isinstance(data, array.array):
        if dtype not in _TYPECODE or data.itemsize != DTYPE_SIZE[dtype] or data.typecode != _TYPECODE[dtype]:
            raise ValueError(f"array of typecode {data.typecode!r} does not hold {dtype}")
        if sys.byteorder == "big":
            data = array.array(data.typecode, data)
            data.byteswap()
        return data.tobytes()
    raw = bytes(data)
    if len(raw) % DTYPE_SIZE[dtype]:
        raise ValueError(f"{len(raw)} bytes is not a whole number of {dtype} items")
    return raw


def encode_record(kind: str, meta: dict, arrays: dict | None = None) -> bytes:
    """One record, ready to append.  `arrays`: {name: (dtype, array.array or bytes)}."""
    if kind not in KINDS:
        raise ValueError(f"unknown record kind {kind!r}")
    if "arrays" in meta:
        raise ValueError("meta must not use the key 'arrays' (the format keeps it)")
    specs, blobs = [], []
    for name, (dtype, data) in (arrays or {}).items():
        if dtype not in DTYPE_SIZE:
            raise ValueError(f"unknown dtype {dtype!r}")
        raw = _le_bytes(dtype, data)
        specs.append([name, dtype, len(raw) // DTYPE_SIZE[dtype]])
        blobs.append(raw)
    mb = json.dumps({**meta, "arrays": specs}, separators=(",", ":")).encode("utf-8")
    body = bytes([KINDS[kind]]) + _U32.pack(len(mb)) + mb + b"".join(blobs)
    return _REC.pack(len(body), zlib.crc32(body)) + body


def decode_body(body: bytes) -> tuple[str, dict, dict]:
    """(kind, meta, arrays) of one record body.  arrays: {name: array.array for i4/f4, bytes for f2}."""
    kind = CODE_KIND.get(body[0])
    if kind is None:
        raise ValueError(f"unknown record kind code {body[0]}")
    (n,) = _U32.unpack_from(body, 1)
    meta = json.loads(body[5:5 + n].decode("utf-8"))
    pos, arrays = 5 + n, {}
    for name, dtype, count in meta.pop("arrays", []):
        size = DTYPE_SIZE[dtype] * count
        raw = body[pos:pos + size]
        if len(raw) != size:
            raise ValueError("record array is cut short")
        pos += size
        if dtype in _TYPECODE:
            a = array.array(_TYPECODE[dtype], raw)
            if sys.byteorder == "big":
                a.byteswap()
            arrays[name] = a
        else:
            arrays[name] = raw
    return kind, meta, arrays


def read_records(f):
    """Yields (kind, meta, arrays) for each whole record after the header of an open segment file; stops at a torn one."""
    while True:
        head = f.read(_REC.size)
        if len(head) < _REC.size:
            return
        n, crc = _REC.unpack(head)
        body = f.read(n)
        if len(body) != n or not body or zlib.crc32(body) != crc:
            return
        try:
            yield decode_body(body)
        except (ValueError, KeyError, IndexError, struct.error):
            return
