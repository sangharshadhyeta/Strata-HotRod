"""memtrace.py - where the memory goes while a Strata server runs, one row per second (written 10 Oct 2026, after two uncapped runs
froze this 31 GB box).  Used by strata_ab.py; the server runs in a systemd scope named `unit` so its cgroup can be read.

Columns (MB unless noted): the system (MemAvailable, MemFree, page cache, swap in use, Mlocked), the scope's cgroup (current,
anonymous, file cache, mapped file, shared, unevictable, major faults, and the kernel's events: high, max, oom, oom_kill), and the
engine process (resident, anonymous and file-backed shares, locked = pinned pages, swapped out).
"""
import csv, os, subprocess, threading, time

COLS = ["t_s", "sys_avail", "sys_free", "sys_cached", "sys_swap_used", "sys_mlocked", "cg_current", "cg_anon", "cg_file",
        "cg_mapped", "cg_shmem", "cg_unevictable", "cg_pgmajfault", "ev_high", "ev_max", "ev_oom", "ev_oom_kill",
        "eng_rss", "eng_anon", "eng_file", "eng_locked", "eng_swap"]


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _kv(text):
    d = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                d[p[0].rstrip(":")] = int(p[1])
            except ValueError:
                pass
    return d


def _children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def find_engine(root_pid):
    stack, seen = [root_pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if _read(f"/proc/{p}/comm").strip() == "strata":
            return p
        stack += _children(p)
    return None


def sample(unit, root_pid, t0):
    cg = f"/sys/fs/cgroup/system.slice/{unit}.scope"
    mi = _kv(_read("/proc/meminfo"))                      # kB
    st = _kv(_read(cg + "/memory.stat"))                  # bytes
    ev = _kv(_read(cg + "/memory.events"))
    cur = int(_read(cg + "/memory.current").strip() or 0)
    eng = find_engine(root_pid)
    sm = _kv(_read(f"/proc/{eng}/smaps_rollup")) if eng else {}   # kB
    mb = lambda b: round(b / 1048576, 1)
    kb = lambda k: round(k / 1024, 1)
    return {"t_s": round(time.time() - t0, 1), "sys_avail": kb(mi.get("MemAvailable", 0)), "sys_free": kb(mi.get("MemFree", 0)),
            "sys_cached": kb(mi.get("Cached", 0)), "sys_swap_used": kb(mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)),
            "sys_mlocked": kb(mi.get("Mlocked", 0)), "cg_current": mb(cur), "cg_anon": mb(st.get("anon", 0)),
            "cg_file": mb(st.get("file", 0)), "cg_mapped": mb(st.get("file_mapped", 0)), "cg_shmem": mb(st.get("shmem", 0)),
            "cg_unevictable": mb(st.get("unevictable", 0)), "cg_pgmajfault": st.get("pgmajfault", 0),
            "ev_high": ev.get("high", 0), "ev_max": ev.get("max", 0), "ev_oom": ev.get("oom", 0), "ev_oom_kill": ev.get("oom_kill", 0),
            "eng_rss": kb(sm.get("Rss", 0)), "eng_anon": kb(sm.get("Anonymous", 0)),
            "eng_file": kb(sm.get("Rss", 0) - sm.get("Anonymous", 0)), "eng_locked": kb(sm.get("Locked", 0)),
            "eng_swap": kb(sm.get("Swap", 0))}


class MemTrace:
    """Samples every `period` seconds into `csv_path` until stop(); peaks() summarises."""

    def __init__(self, unit, root_pid, csv_path, period=1.0):
        self.unit, self.root_pid, self.path, self.period = unit, root_pid, csv_path, period
        self.rows, self._stop, self.t0 = [], threading.Event(), time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            while not self._stop.is_set():
                try:
                    r = sample(self.unit, self.root_pid, self.t0)
                    self.rows.append(r)
                    w.writerow(r)
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=10)

    def peaks(self):
        if not self.rows:
            return {}
        top = lambda k: max(r[k] for r in self.rows)
        last = self.rows[-1]
        return {"samples": len(self.rows), "peak_cg_current_mb": top("cg_current"), "peak_cg_anon_mb": top("cg_anon"),
                "peak_cg_file_mb": top("cg_file"), "peak_eng_rss_mb": top("eng_rss"), "peak_eng_locked_mb": top("eng_locked"),
                "peak_eng_swap_mb": top("eng_swap"), "peak_sys_swap_used_mb": top("sys_swap_used"),
                "min_sys_avail_mb": min(r["sys_avail"] for r in self.rows), "min_sys_free_mb": min(r["sys_free"] for r in self.rows),
                "events_at_end": {k: last[k] for k in ("ev_high", "ev_max", "ev_oom", "ev_oom_kill")},
                "cg_pgmajfault_at_end": last["cg_pgmajfault"]}
