"""capture_read.py - reads the capture ring (serve/capture.py) for the idle trainer.

    from capture_read import iter_records, preference_pairs
    for rec in iter_records("/mnt/nvme/strata-data/capture", kinds={"A"}, fingerprint=fp):
        rec["meta"], rec["arrays"]["prompt"], rec["arrays"]["answer"]      # arrays: array('i') of token ids

A segment whose header is not ours (or another format version) raises ValueError, unless it is skipped with skip_bad=True.
A segment ends at its last whole record: a torn tail (a crash mid-write) is simply not read.  Read-only: nothing is changed.
"""
import os

try:
    from .capture_format import read_header, read_records
except ImportError:          # run as a script or imported with serve/ on sys.path
    from capture_format import read_header, read_records

SEG_PREFIX, SEG_SUFFIX = "seg-", ".cap"


def segment_paths(directory: str) -> list[str]:
    """The segment files of the ring, oldest first."""
    found = []
    for name in os.listdir(directory):
        if name.startswith(SEG_PREFIX) and name.endswith(SEG_SUFFIX):
            digits = name[len(SEG_PREFIX):-len(SEG_SUFFIX)]
            if digits.isdigit():
                found.append((int(digits), os.path.join(directory, name)))
    return [p for _, p in sorted(found)]


def read_segment_header(path: str) -> dict:
    with open(path, "rb") as f:
        return read_header(f)


def iter_records(directory: str, kinds=None, fingerprint: str | None = None, skip_bad: bool = False):
    """Yields {"kind", "meta", "arrays", "segment"} for every whole record, oldest first.  `kinds`: e.g. {"A", "B"} (None: all);
    `fingerprint`: only segments written by that engine build and setting (the prompt cache's fingerprint)."""
    for path in segment_paths(directory):
        try:
            with open(path, "rb") as f:
                header = read_header(f)
                if fingerprint is not None and header.get("fingerprint") != fingerprint:
                    continue
                for kind, meta, arrays in read_records(f):
                    if kinds is not None and kind not in kinds:
                        continue
                    yield {"kind": kind, "meta": meta, "arrays": arrays, "segment": path}
        except ValueError:
            if not skip_bad:
                raise


def preference_pairs(directory: str, fingerprint: str | None = None) -> list[dict]:
    """B joined into preference pairs: {"chosen": id, "rejected": id, "conv", "t"} for each regenerate pair (a "kept" event names the
    answer kept and the one it replaced), and {"request": id, "chosen": bool, "t"} for each explicit preference as given."""
    pairs = []
    for rec in iter_records(directory, kinds={"B"}, fingerprint=fingerprint, skip_bad=True):
        meta = rec["meta"]
        if meta.get("type") == "kept":
            pairs.append({"chosen": meta["req"], "rejected": meta["over"], "conv": meta.get("conv"), "t": meta.get("t")})
        elif meta.get("type") == "explicit":
            pairs.append({"request": meta["req"], "chosen": meta["chosen"], "t": meta.get("t")})
    return pairs
