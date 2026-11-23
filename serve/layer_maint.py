"""Keeps the per-layer threshold table (<pack>/layer-sens.txt) fresh from the engine's own live capture, with nobody running a tool.

The engine samples a few decode windows into small segment files (<pack>/layer-capture/cap-*.bin, at most 8 waiting) and reads <pack>/layer-sens.txt whenever it appears or
changes.  Between them this module runs tools/layer_table.py: when the server has been idle for a while, enough segments are waiting and the machine has the memory and the
CPU to spare, it measures a slice of the layers (each run a different slice, so the whole model comes round in a few runs), blends the result into the table (the tool keeps an
exponential average, so the table keeps improving with use), and deletes the segments it used so the engine can capture more.  The job runs at the lowest CPU and disk priority
in its own process, one thread, and never while a request is queued or running.  STRATA_LAYER_MAINT=0 turns it off; it only runs when the capture is on (STRATA_LAYER_CAPTURE=1).
The live per-layer thresholds in the engine do not need any of this: they come from the router weights it sees (layer_live in expert_source.cpp); this is the optional
refinement from exact expert outputs.
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time


def gguf_glob(native):
    """The glob of all the shard files of the model named by --native ('x-00001-of-00004.gguf' -> 'x-*-of-00004.gguf'); a single file is itself."""
    base = os.path.basename(native)
    if "-of-" in base and base.endswith(".gguf"):
        head, tail = base.split("-of-", 1)
        stem = head.rsplit("-", 1)[0]
        return os.path.join(os.path.dirname(native), stem + "-*-of-" + tail)
    return native


def mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def segments(cap_dir, min_age_s=3.0, now=None):
    """The complete segment files waiting (the engine writes cap-N.bin.tmp and renames it when the segment is whole)."""
    now = time.time() if now is None else now
    out = []
    for p in sorted(glob.glob(os.path.join(cap_dir, "cap-*.bin"))):
        try:
            if now - os.path.getmtime(p) >= min_age_s:
                out.append(p)
        except OSError:
            pass
    return out


def should_run(now, last_request_at, busy, last_run, n_segs, mem_gib, load1, ncpu, idle_s=90.0, min_gap_s=240.0, min_segs=4, min_mem_gib=2.5, max_load=0.7):
    """(run?, why not).  Pure, so it can be tested: idle for idle_s with nothing queued or running, min_segs segments waiting, min_gap_s since the last run,
    min_mem_gib of memory available, and the machine not busy (1-minute load under max_load per CPU)."""
    if busy:
        return False, "a request is running or queued"
    if last_request_at is not None and now - last_request_at < idle_s:
        return False, "not idle yet"
    if now - last_run < min_gap_s:
        return False, "ran recently"
    if n_segs < min_segs:
        return False, "%d of %d segments" % (n_segs, min_segs)
    if mem_gib < min_mem_gib:
        return False, "only %.1f GiB memory available" % mem_gib
    if load1 > max_load * max(ncpu, 1):
        return False, "the machine is busy"
    return True, ""


def build_command(tools_dir, segs, gguf, out, rows, part, parts, threads=1):
    """The refresh job: lowest CPU and disk priority, one slice of the layers."""
    cmd = ["nice", "-n", "19"]
    if os.path.exists("/usr/bin/ionice"):
        cmd += ["ionice", "-c", "3"]
    cmd += [sys.executable, os.path.join(tools_dir, "layer_table.py"), "--trace"] + list(segs)
    cmd += ["--gguf", gguf, "--out", out, "--rows", str(rows), "--min-rows", "40", "--part", "%d/%d" % (part, parts), "--threads", str(threads), "--layers", "0"]
    return cmd


class LayerMaint:
    def __init__(self, sens_path, native, tools_dir, busy, last_request, log=print, parts=4, rows=150, run=None, now=time.time, **gates):
        self.sens_path, self.native, self.tools_dir = sens_path, native, tools_dir
        self.cap_dir = os.path.join(os.path.dirname(sens_path), "layer-capture")
        self.state_path = sens_path + ".state"
        self.busy, self.last_request, self.log = busy, last_request, log
        self.parts, self.rows, self.gates = parts, rows, gates
        self._run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=1800))
        self._now = now
        self.last_run = 0.0
        self.fail_streak = 0
        self.stop = False

    def _state(self):
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"part": 0, "runs": 0}

    def _save(self, st):
        tmp = self.state_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def tick(self):
        """One look: run the refresh job if the conditions hold.  Returns a short description of what it did, or None."""
        now = self._now()
        segs = segments(self.cap_dir, now=now)
        gap = self.gates.get("min_gap_s", 240.0) * (2 ** min(self.fail_streak, 4))   # a failing job backs off
        ok, why = should_run(now, self.last_request(), self.busy(), self.last_run, len(segs), mem_available_gib(), os.getloadavg()[0], os.cpu_count() or 1,
                             **{**self.gates, "min_gap_s": gap})
        if not ok:
            return None
        st = self._state()
        part = int(st.get("part", 0)) % self.parts
        self.last_run = now
        cmd = build_command(self.tools_dir, segs, gguf_glob(self.native), self.sens_path, self.rows, part, self.parts)
        t0 = time.time()
        try:
            r = self._run(cmd)
            good = r.returncode == 0
            tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
            err = (r.stderr or "").strip()[-200:]
        except (OSError, subprocess.SubprocessError) as e:
            good, tail, err = False, [""], str(e)
        if good:
            for p in segs:                                  # consumed: the engine may capture more
                try:
                    os.remove(p)
                except OSError:
                    pass
            self.fail_streak = 0
            st = {"part": (part + 1) % self.parts, "runs": int(st.get("runs", 0)) + 1, "last": int(now)}
            self._save(st)
            msg = "layer table refreshed (part %d/%d, %d segments, %.0f s): %s" % (part, self.parts, len(segs), time.time() - t0, tail[0])
        else:
            self.fail_streak += 1
            msg = "layer table refresh failed (part %d/%d): %s" % (part, self.parts, err or "no output")
        self.log("[strata] " + msg)
        return msg

    def start(self):
        def loop():
            while not self.stop:
                time.sleep(30.0)
                try:
                    self.tick()
                except Exception as e:                      # the job must never take the server down
                    self.log("[strata] layer table refresh: %s" % e)
        threading.Thread(target=loop, daemon=True, name="layer-maint").start()
        self.log("[strata] layer table: refreshed automatically from the live capture when idle (%s)" % self.sens_path)


def from_engine_args(args, env, tools_dir, busy, last_request, log=print):
    """A LayerMaint for the engine started with `args` and `env` (the config's), or None when there is nothing to maintain."""
    e = dict(os.environ)
    e.update(env or {})
    if e.get("STRATA_LAYER_MAINT") == "0" or e.get("STRATA_LAYER_CAPTURE") != "1":   # nothing to refresh from unless the capture was asked for
        return None
    pairs = {k: v for k, v in zip(args, args[1:])}
    native = pairs.get("--native")
    sens = e.get("STRATA_LAYER_SENS")
    if not sens:
        prof = pairs.get("--expert-profile") or pairs.get("--expert-profile-save")
        if not prof:
            return None
        sens = os.path.join(os.path.dirname(os.path.abspath(prof)), "layer-sens.txt")
    if not native or not sens:
        return None
    return LayerMaint(sens, native, tools_dir, busy, last_request, log=log)
