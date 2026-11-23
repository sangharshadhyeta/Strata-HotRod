"""Keeps the low-rank copy sidecar (<pack>/lowrank.bin) fitted for the experts the engine skips most, with nobody running a tool - and stops when it has settled.

The engine counts, per layer and expert, how often the weight rule skipped a missed expert (<pack>/layer-skips.bin).  While the server is idle this module runs tools/lowrank_fit.py
for a small batch of the most-skipped experts that have no copy yet.  The sidecar SETTLES when the copies cover the target share of every layer's skipped picks, or when three
batches in a row add nothing: from then on the job does no fitting and only looks at the counts once an hour.  If the traffic shifts (the share the copies cover drops below
WAKE_COVER, e.g. because other experts are being skipped now) it wakes and fits the newly skipped ones.  Opt-in: STRATA_LOWRANK=1, with STRATA_LOWRANK_LAYERS=<comma list>
(the layers the offline survey says a copy pays in); STRATA_LOWRANK_RANK (default 128).  Needs the live thresholds (they feed the counts)."""
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from serve.layer_maint import gguf_glob, mem_available_gib, should_run, segments   # noqa: E402

TARGET_COVER = 0.90      # a layer is covered when its copies serve this share of the picks it skips
WAKE_COVER = 0.80        # a settled sidecar wakes when some layer falls below this
CALM_RUNS = 3            # batches in a row that add nothing
SETTLED_CHECK_S = 3600.0


def coverage(skips, have, layers):
    """{layer: share of that layer's skipped picks served by experts that have a copy} (1.0 for a layer nothing was skipped in)."""
    out = {}
    for l in layers:
        c = skips.get(l)
        total = float(c.sum()) if c is not None else 0.0
        out[l] = 1.0 if total <= 0 else sum(float(c[e]) for e in have.get(l, ())) / total
    return out


def next_state(st, cover, added):
    """The new (settled, calm) after a batch that added `added` copies and left the layers at `cover`."""
    calm = 0 if added > 0 else st.get("calm", 0) + 1
    done = all(v >= TARGET_COVER for v in cover.values())
    return (done or calm >= CALM_RUNS), calm


def wake(st_settled, cover):
    """A settled sidecar wakes when any layer's covered share fell below WAKE_COVER."""
    return st_settled and any(v < WAKE_COVER for v in cover.values())


class LowrankMaint:
    def __init__(self, out_path, native, tools_dir, layers, busy, last_request, log=print, rank=128, run=None, now=time.time, **gates):
        self.out, self.native, self.tools_dir, self.layers, self.rank = out_path, native, tools_dir, list(layers), rank
        self.skips_path = os.path.join(os.path.dirname(out_path), "layer-skips.bin")
        self.cap_dir = os.path.join(os.path.dirname(out_path), "layer-capture")
        self.state_path = out_path + ".state"
        self.busy, self.last_request, self.log, self.gates = busy, last_request, log, gates
        self._run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=3600))
        self._now = now
        self.last_run = 0.0
        self.last_check = 0.0
        self.fail_streak = 0
        self.stop = False

    def _state(self):
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"settled": False, "calm": 0, "runs": 0}

    def _save(self, st):
        tmp = self.state_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def _cover(self):
        import lowrank_fit
        have = {}
        for l, e, r in lowrank_fit.read_records(self.out):
            have.setdefault(l, set()).add(e)
        return coverage(lowrank_fit.read_skips(self.skips_path), have, self.layers)

    def tick(self):
        now = self._now()
        st = self._state()
        if not os.path.exists(self.skips_path):
            return None
        if st.get("settled"):
            if now - self.last_check < SETTLED_CHECK_S:
                return None                                  # settled: nothing is asked of the machine between the hourly looks
            self.last_check = now
            try:
                cov = self._cover()
            except (OSError, ValueError):
                return None
            if not wake(True, cov):
                return None
            st.update(settled=False, calm=0)
            self._save(st)
            msg = "low-rank copies: the traffic shifted (coverage %s), fitting again" % " ".join("%d:%.2f" % kv for kv in cov.items())
            self.log("[strata] " + msg)
            return msg
        gap = self.gates.get("min_gap_s", 600.0) * (2 ** min(self.fail_streak, 4))
        ok, _ = should_run(now, self.last_request(), self.busy(), self.last_run, 10 ** 6, mem_available_gib(), os.getloadavg()[0], os.cpu_count() or 1,
                           **{**self.gates, "min_gap_s": gap, "min_segs": 0})
        if not ok:
            return None
        segs = segments(self.cap_dir, now=now)
        if not segs:
            return None                                       # the fit needs router inputs: the live capture's segments (STRATA_LAYER_CAPTURE=1)
        self.last_run = now
        cmd = ["nice", "-n", "19"] + (["ionice", "-c", "3"] if os.path.exists("/usr/bin/ionice") else []) + [
            sys.executable, os.path.join(self.tools_dir, "lowrank_fit.py"), "--gguf", gguf_glob(self.native), "--trace"] + segs + [
            "--skips", self.skips_path, "--out", self.out, "--layers", ",".join(str(v) for v in self.layers), "--rank", str(self.rank),
            "--batch", "24", "--cover", str(TARGET_COVER), "--threads", "1"]
        t0 = time.time()
        try:
            r = self._run(cmd)
            good, out, err = r.returncode == 0, (r.stdout or ""), (r.stderr or "").strip()[-200:]
        except (OSError, subprocess.SubprocessError) as e:
            good, out, err = False, "", str(e)
        if not good:
            self.fail_streak += 1
            msg = "low-rank copies: the fit failed: %s" % (err or "no output")
            self.log("[strata] " + msg)
            return msg
        self.fail_streak = 0
        cover, added = {}, 0
        for ln in out.splitlines():
            if ln.startswith("COVERAGE"):
                head, tail = ln.split("|", 1)[0], ln.split("|", 2)
                for kv in head.split()[1:]:
                    l, v = kv.split(":")
                    cover[int(l)] = float(v)
                added = int(tail[1].split()[1]) if len(tail) > 1 and tail[1].split() else 0
        settled, calm = next_state(st, cover, added)
        st.update(settled=settled, calm=calm, runs=st.get("runs", 0) + 1, last=int(now))
        self._save(st)
        msg = "low-rank copies: +%d (%.0f s), coverage %s%s" % (added, time.time() - t0, " ".join("%d:%.2f" % kv for kv in sorted(cover.items())),
                                                             "; settled: no more fitting until the traffic shifts" if settled else "")
        self.log("[strata] " + msg)
        return msg

    def start(self):
        def loop():
            while not self.stop:
                time.sleep(60.0)
                try:
                    self.tick()
                except Exception as e:
                    self.log("[strata] low-rank copies: %s" % e)
        threading.Thread(target=loop, daemon=True, name="lowrank-maint").start()
        self.log("[strata] low-rank copies: fitted for the most-skipped experts of layers %s in idle time (%s)" % (self.layers, self.out))


def from_engine_args(args, env, tools_dir, busy, last_request, log=print):
    e = dict(os.environ)
    e.update(env or {})
    if e.get("STRATA_LOWRANK") != "1" or e.get("STRATA_LAYER_CAPTURE") != "1":
        return None
    pairs = {k: v for k, v in zip(args, args[1:])}
    native = pairs.get("--native")
    prof = pairs.get("--expert-profile") or pairs.get("--expert-profile-save")
    layers = [int(v) for v in e.get("STRATA_LOWRANK_LAYERS", "").split(",") if v.strip().isdigit()]
    if not native or not prof or not layers:
        return None
    return LowrankMaint(os.path.join(os.path.dirname(os.path.abspath(prof)), "lowrank.bin"), native, tools_dir, layers, busy, last_request, log=log,
                        rank=int(e.get("STRATA_LOWRANK_RANK", "128")))
