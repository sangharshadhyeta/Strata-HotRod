"""capture.py - the capture record of the living model (docs/SPEC-capture.md): sections A and B on the server side, the bounded
ring of segment files and its one writer thread, and the hook the engine will call for its records C, D and E (a later step).

Opt-in: nothing here runs unless the config has capture on (STRATA_CAPTURE=1 in the environment, or "capture": true; settings()).
With it off no directory is made, no thread starts and nothing is written.

The request path only ever calls put_nowait on a bounded queue: a full queue DROPS the record and counts it ("K dropped" on the
request line), it never waits.  The writer thread does the rest - the conversation key, the regenerate tracking, the encoding and
the disk - so a request pays for one tuple.

Regenerate detection (B) is token-id equality only: two requests of one conversation (the same conversation key, the tokens up to
the third <|im_start|>) with equal prompt ids -> the earlier answer is "replaced"; the answer a later prompt contains (as a run of
ids) is "kept", and the kept/replaced pair is a preference pair.  Records hold token ids, never text.
"""
import array
import collections
import hashlib
import os
import queue
import shutil
import sys
import threading
import time
import uuid

try:
    from .capture_format import encode_header, encode_record
    from .prompt_cache import conversation_len
except ImportError:          # imported as a top-level module (serve/ on sys.path, as server.py does)
    from capture_format import encode_header, encode_record
    from prompt_cache import conversation_len

DEFAULT_DIR = "/mnt/nvme/strata-data/capture"
DEFAULT_GIB = 4.0            # capture_gib: at most this many GiB, and never more than 1/16 of the drive's free space
DEFAULT_AGE_D = 14.0         # capture_max_age_d: a segment not written for longer goes (0: no age limit)
QUEUE_MAX = 256              # records waiting for the writer; a full queue drops the next one
MIN_SEG = 64 << 10           # a segment is an eighth of the ring, between these bounds
MAX_SEG = 64 << 20
SEG_PREFIX, SEG_SUFFIX = "seg-", ".cap"
ENGINE_KINDS = ("C", "D", "E")
CONVS_KEPT = 64              # regenerate tracking: conversations kept in memory, and answers kept per conversation
ANSWERS_KEPT = 16


def capture_dir(value, base: str | None = None) -> str:
    """The capture folder as one absolute path (relative to `base`, else the working directory), created private to this user.
    ValueError for anything else (the same rule as the slot save folder)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("capture_path must be a directory path (a string)")
    if any(ord(c) < 32 for c in value):
        raise ValueError("capture_path must not contain control characters")
    path = os.path.abspath(os.path.join(base, value) if base and not os.path.isabs(value) else value)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not os.path.isdir(path):
        raise ValueError(f"capture_path {path} is not a directory")
    return path


def settings(cfg: dict, env) -> dict | None:
    """The capture settings, or None when capture is off (STRATA_CAPTURE unset and "capture" not true).  ValueError for a bad value."""
    on_cfg = cfg.get("capture", False)
    if not isinstance(on_cfg, bool):
        raise ValueError('config "capture" must be true or false')
    if not (on_cfg or env.get("STRATA_CAPTURE") == "1"):
        return None
    gib = cfg.get("capture_gib", DEFAULT_GIB)
    if isinstance(gib, bool) or not isinstance(gib, (int, float)) or gib <= 0:
        raise ValueError('config "capture_gib" must be a number of GiB above 0')
    age = cfg.get("capture_max_age_d", DEFAULT_AGE_D)
    if isinstance(age, bool) or not isinstance(age, (int, float)) or age < 0:
        raise ValueError('config "capture_max_age_d" must be a number of days (0: no age limit)')
    return {"dir": capture_dir(cfg.get("capture_path") or DEFAULT_DIR, cfg.get("cwd")), "gib": float(gib),
            "max_age_d": float(age)}


def _ids_contain(hay: bytes, needle: bytes) -> bool:
    """True when the id run `needle` occurs in `hay` (both native int32 bytes), aligned to whole ids.  Id equality only."""
    if not needle:
        return False
    i = hay.find(needle)
    while i != -1:
        if i % 4 == 0:
            return True
        i = hay.find(needle, i + 1)
    return False


class Regenerates:
    """B: regenerate detection within each conversation, kept in memory (the writer thread only)."""

    def __init__(self):
        self.convs: collections.OrderedDict[str, dict] = collections.OrderedDict()

    def observe(self, rid: str, conv: str | None, prompt_b: bytes, answer_b: bytes) -> list[dict]:
        """Events for one finished request: {"type": "kept", "req", "over"} when a pending pair's answer is contained in this
        prompt, {"type": "replaced", "req", "by"} when this prompt equals an earlier one's (that earlier answer is replaced)."""
        if conv is None:
            return []
        st = self.convs.pop(conv, None) or {"entries": collections.deque(maxlen=ANSWERS_KEPT), "pending": []}
        self.convs[conv] = st
        while len(self.convs) > CONVS_KEPT:
            self.convs.popitem(last=False)
        events, still = [], []
        for pair in st["pending"]:
            kept = next((m for m in pair if _ids_contain(prompt_b, m["answer"])), None)
            if kept is None:
                still.append(pair)
                continue
            over = pair[1] if kept is pair[0] else pair[0]
            events.append({"type": "kept", "req": kept["req"], "over": over["req"], "conv": conv})
        st["pending"] = still
        digest = hashlib.sha1(prompt_b).digest()
        me = {"req": rid, "answer": answer_b}
        for e in st["entries"]:
            if not e["replaced"] and e["digest"] == digest:
                e["replaced"] = True
                events.append({"type": "replaced", "req": e["m"]["req"], "by": rid, "conv": conv})
                st["pending"].append((e["m"], me))
        st["entries"].append({"digest": digest, "m": me, "replaced": False})
        return events


class Capture:
    """The ring and its writer.  `start=False` leaves the writer thread off (tests: the queue fills and drops)."""

    def __init__(self, directory: str, fingerprint: str, model: str = "", gib: float | None = DEFAULT_GIB,
                 max_age_d: float = DEFAULT_AGE_D, turn_id: int | None = None, queue_max: int = QUEUE_MAX,
                 start: bool = True, cap_bytes: int | None = None):
        self.dir = os.path.abspath(directory)
        self.fingerprint, self.model = fingerprint, model
        self.gib = gib
        self.max_age_s = float(max_age_d) * 86400.0
        self.turn_id = turn_id
        self.cap_bytes = cap_bytes              # tests: a fixed cap in place of the free-space rule
        self.session = uuid.uuid4().hex[:8]     # the request ids: <session>-<n>, unique across restarts
        self.lock = threading.Lock()
        self.written = 0                        # bytes appended to the ring since this process started
        self.dropped = 0                        # records not written: the queue was full, or a write failed or did not fit
        self.records = 0
        self.errors = 0
        self.regen = Regenerates()
        self.q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._n = 0
        self._fd = None
        self._active = None
        self._active_size = 0
        self._seg_bytes = MIN_SEG
        self._thread = None
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        # owner-only, as the prompt cache's folder: makedirs does not change a folder that exists, and a file made with open() takes
        # the umask, so both are set here; segments of an earlier run are tightened too
        try:
            os.chmod(self.dir, 0o700)
            for seq, path, _, _ in self._segments():
                os.chmod(path, 0o600)
        except OSError:
            pass
        segs = self._segments()
        self._seq = (segs[-1][0] + 1) if segs else 1
        self._trim(reserve=0)                   # the age and size rules on what is already here
        if start:
            self.start()

    # ---- the ring on disk
    def _segments(self) -> list[tuple[int, str, int, float]]:
        """(seq, path, bytes, mtime) of every segment file in the folder, oldest first."""
        out = []
        for name in os.listdir(self.dir):
            if not (name.startswith(SEG_PREFIX) and name.endswith(SEG_SUFFIX)):
                continue
            digits = name[len(SEG_PREFIX):-len(SEG_SUFFIX)]
            if not digits.isdigit():
                continue
            path = os.path.join(self.dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            out.append((int(digits), path, st.st_size, st.st_mtime))
        return sorted(out)

    def _cap(self) -> int:
        """The ring's size limit now: capture_gib, and never more than an sixteenth of the drive's free space."""
        if self.cap_bytes is not None:
            return self.cap_bytes
        cap = shutil.disk_usage(self.dir).free // 16
        if self.gib:
            cap = min(cap, int(self.gib * (1 << 30)))
        return cap

    def _remove(self, path: str):
        try:
            os.remove(path)
        except OSError:
            pass

    def _trim(self, reserve: int):
        """Drops the segments past max_age_s, then the oldest until the closed ones plus `reserve` fit under the limit.  The segment
        being written is never touched (it is not in the closed list)."""
        cap = self._cap()
        now = time.time()
        closed = [s for s in self._segments() if s[1] != self._active]
        if self.max_age_s:
            kept = []
            for s in closed:
                if now - s[3] > self.max_age_s:
                    self._remove(s[1])
                else:
                    kept.append(s)
            closed = kept
        total = sum(s[2] for s in closed)
        for _, path, size, _ in closed:          # oldest first
            if total + reserve <= cap:
                break
            self._remove(path)
            total -= size

    def _write(self, data: bytes):
        view = memoryview(data)
        while view:
            n = os.write(self._fd, view)
            view = view[n:]
        self._active_size += len(data)
        with self.lock:
            self.written += len(data)

    def _close_segment(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _rotate(self) -> bool:
        """Starts a new segment after trimming the ring so that it, too, fits under the limit.  False when the limit is below the
        smallest segment (the free space is gone): the record is then dropped."""
        self._close_segment()
        self._active = None
        cap = self._cap()
        if cap < MIN_SEG:
            return False
        seg = max(MIN_SEG, min(MAX_SEG, cap // 8))
        self._trim(reserve=seg)
        path = os.path.join(self.dir, f"{SEG_PREFIX}{self._seq:08d}{SEG_SUFFIX}")
        self._seq += 1
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self._active, self._seg_bytes, self._active_size = path, seg, 0
        self._write(encode_header(self.fingerprint, self.model or "", self._seq - 1, time.time()))
        return True

    def _append(self, rec: bytes) -> bool:
        if len(rec) + 4096 > MAX_SEG:            # no segment could hold it
            return False
        if self._fd is None or self._active_size + len(rec) > self._seg_bytes:
            if not self._rotate():
                return False
        if self._active_size + len(rec) > self._seg_bytes:
            return False
        self._write(rec)
        return True

    # ---- the writer thread
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="capture-writer", daemon=True)
            self._thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                for rec in self._encode(item):
                    if self._append(rec):
                        with self.lock:
                            self.records += 1
                    else:
                        self._count_drop()
            except Exception as e:               # a record that fails is dropped and counted; the writer goes on
                with self.lock:
                    self.errors += 1
                self._count_drop()
                if self.errors == 1:
                    print(f"[strata] capture: a record was dropped ({e!r}); later ones are counted, not printed", flush=True)
            finally:
                self.q.task_done()

    def _new_id(self) -> str:
        self._n += 1
        return f"{self.session}-{self._n:08d}"

    def _encode(self, item) -> list[bytes]:
        """The records one queued item becomes (writer thread only: the conversation key, the regenerate events, the encoding)."""
        kind = item[0]
        if kind == "A":
            _, ids, answer, finish, extra = item
            rid = self._new_id()
            clen = conversation_len(ids, self.turn_id)
            conv = hashlib.sha1(array.array("i", ids[:clen]).tobytes()).hexdigest()[:16] if clen else None
            p, a = array.array("i", ids), array.array("i", answer)
            meta = {"req": rid, "t": round(time.time(), 3), "conv": conv, "conv_len": clen or 0, "finish": finish,
                    "n_prompt": len(ids), "n_answer": len(answer), **extra}
            out = [encode_record("A", meta, {"prompt": ("i4", p), "answer": ("i4", a)})]
            for e in self.regen.observe(rid, conv, p.tobytes(), a.tobytes()):
                out.append(encode_record("B", {**e, "t": round(time.time(), 3)}))
            return out
        if kind == "P":
            _, rid, chosen = item
            return [encode_record("B", {"type": "explicit", "req": rid, "chosen": chosen, "t": round(time.time(), 3)})]
        if kind == "E":
            _, ekind, meta, arrays = item
            return [encode_record(ekind, {"t": round(time.time(), 3), **meta}, arrays)]
        raise ValueError(f"unknown queue item {kind!r}")

    # ---- the request path (never waits)
    def _count_drop(self):
        with self.lock:
            self.dropped += 1

    def _submit(self, item) -> bool:
        try:
            self.q.put_nowait(item)
            return True
        except queue.Full:
            self._count_drop()
            return False

    def request(self, ids, answer, finish: str, done: dict) -> bool:
        """Record A for one finished engine pass: its prompt ids, the ids it generated, the finish reason and the engine's counters
        from the `done` event.  Returns False when the record was dropped (the queue was full)."""
        extra = {k: done[k] for k in ("completion_tokens", "reasoning_tokens", "reasoning_recoveries", "reused")
                 if isinstance(done.get(k), int) and not isinstance(done.get(k), bool)}
        return self._submit(("A", list(ids), list(answer), str(finish), extra))

    def add_preference(self, request_id: str, chosen: bool) -> bool:
        """B, explicit: a client's preference for a request ({"request": id, "chosen": true|false}), recorded as given."""
        return self._submit(("P", str(request_id), bool(chosen)))

    def add_engine_record(self, kind: str, meta: dict, arrays: dict | None = None) -> bool:
        """THE HOOK for the engine's records (SPEC sections C, D, E): kind "C", "D" or "E", a meta dict, and the arrays as
        {name: (dtype, array.array or bytes)} with dtype "i4", "f4" or "f2".  The arrays are queued as given: the caller must not change
        them afterwards (copy them first).  Returns False when the queue was full (the record is dropped and counted)."""
        if kind not in ENGINE_KINDS:
            raise ValueError(f"the engine's records are C, D or E, not {kind!r}")
        return self._submit(("E", kind, dict(meta), dict(arrays or {})))

    def status(self) -> str:
        """The request line's counter."""
        with self.lock:
            return f"capture: {self.written >> 10} KB written, {self.dropped} dropped"

    def close(self, timeout: float = 10.0):
        """Writes what is queued, then stops the writer and closes the segment (the server's shutdown, and the tests)."""
        if self._thread is not None:
            try:
                self.q.put(None, timeout=timeout)
            except queue.Full:
                pass
            self._thread.join(timeout)
            self._thread = None
        self._close_segment()


def print_started(c: Capture, gib, age):
    print(f"[strata] capture on: {c.dir} (at most {gib:g} GiB, segments unwritten for {age:g} days go); "
          "the records stay on this PC", file=sys.stdout, flush=True)
