"""Interleaved A/B of Strata server settings on one box, solid.cpp's method: arms alternate (A B A B), the two A runs are the
identical-config control (the noise floor), the model files' page cache is dropped before every start, nothing else runs.

Each arm: start serve/server.py with the arm's environment, wait until the model is loaded, run qwen_tps.py (chat endpoint,
code / reasoning / prose prompts, thinking off), read the per-request "file tier I/O" lines the engine logs (bytes the OS
read from storage against bytes of expert data the engine read), stop the server by exact PID.

usage: strata_ab.py CONFIG_JSON LOGFILE OUT_JSON PORT ARM...   (ARM = NAME:ENV1=V1,ENV2=V2 ; NAME:- for no extra environment)
"""
import json, os, signal, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TPS = "/root/.claude/jobs/743230e4/tmp/qwen_tps.py"
MODEL_DIRS = ["/mnt/nvme/models/qwen38-flash-next/GSQ-RCO-IQ3_S/IQ3_S", "/mnt/nvme/strata-data/packs/iq3_s"]


def drop_cache():
    n = 0
    for d in MODEL_DIRS:
        for dp, _, fs in os.walk(d):
            for f in fs:
                try:
                    fd = os.open(os.path.join(dp, f), os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                    n += 1
                except OSError:
                    pass
    return n


def health(port):
    try:
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read())
    except Exception:
        return {}


def children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def stop(pid):
    pids = [pid] + children(pid)
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + 90
    while time.time() < end and any(os.path.exists(f"/proc/{p}") and open(f"/proc/{p}/stat").read().split()[2] != "Z" for p in pids):
        time.sleep(1)
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(5)


def io_lines(text):
    """(OS MB read, major faults, expert MB read) per request, from the engine's 'file tier I/O this request' lines."""
    rows = []
    for line in text.splitlines():
        if "file tier I/O this request: the OS read" not in line:
            continue
        a = line.split("the OS read ")[1]
        os_mb = float(a.split(" MB from storage")[0])
        faults = int(a.split("(")[1].split(" major faults")[0])
        exp_mb = float(a.split("for ")[1].split(" MB of expert reads")[0])
        rows.append((os_mb, faults, exp_mb))
    return rows


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPEATS = os.environ.get("AB_REPEATS", "3")   # timed requests per prompt kind (AB_REPEATS=1 for a short, watched first run)
MEMORY_CAP_GB = int(os.environ.get("AB_CAP_GB", "24"))   # the server's memory cap (page cache included), no swap; the soft limit is 1 GB under
MIN_AVAILABLE_MB = 1500       # stop the arm when less RAM than this is available
MAX_SWAP_GROWTH_MB = 1500     # or when the swap in use has grown by more than this since the arm began
VMSTAT_KEYS =("pswpin", "pswpout", "pgmajfault", "compact_stall", "compact_fail", "thp_fault_fallback")


def vmstat():
    out = {}
    for line in open("/proc/vmstat"):
        k, _, v = line.partition(" ")
        if k in VMSTAT_KEYS:
            out[k] = int(v)
    return out


def meminfo_mb(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) // 1024
    return None


def sha_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def run_arm(name, env_extra, cfg, log, port):
    engine = json.load(open(cfg)).get("exe", "")
    engine_sha = sha_of(engine) if engine and os.path.exists(engine) else None
    dropped = drop_cache()
    vm0 = vmstat()
    swap_used0 = meminfo_mb("SwapTotal") - meminfo_mb("SwapFree")
    start_size = os.path.getsize(log) if os.path.exists(log) else 0
    env = dict(os.environ)
    env.update(env_extra)
    # The server runs in its own memory-capped systemd scope with NO swap (10 Oct 2026: two uncapped runs froze this 31 GB box
    # and it had to be hard-reset): past the cap the kernel kills the server, not the machine. MEMORY_CAP_GB leaves the rest of
    # the system (the shell, the editor, the kernel) about 7 GB.
    unit = f"strata-ab-{name}-{int(time.time())}"
    cap = ["systemd-run", "--scope", "--quiet", f"--unit={unit}", "-p", f"MemoryMax={MEMORY_CAP_GB}G",
           "-p", f"MemoryHigh={MEMORY_CAP_GB - 1}G", "-p", "MemorySwapMax=0"]
    srv = subprocess.Popen(cap + [ROOT + "/.venv/bin/python", ROOT + "/serve/server.py", "--engine", "strata", "--config", cfg, "--port", str(port)],
                           cwd=ROOT, env=env, stdout=open("/root/.claude/jobs/743230e4/tmp/strata_ab_server.out", "w"), stderr=subprocess.STDOUT)
    from memtrace import MemTrace   # where the memory goes, once a second (tools/linkbench/memtrace.py)
    trace = MemTrace(unit, srv.pid, f"/root/.claude/jobs/743230e4/tmp/memtrace_{name}.csv").start()
    # Memory guard (10 Oct 2026: a run on this 31 GB box thrashed swap until the machine hung and was reset): if the RAM left
    # or the swap grown crosses a line at any moment of the arm, the server is stopped at once and the arm is void.
    guard = {"tripped": None, "done": False}

    def watch():
        while not guard["done"]:
            avail = meminfo_mb("MemAvailable")
            grown = (meminfo_mb("SwapTotal") - meminfo_mb("SwapFree")) - swap_used0
            if avail < MIN_AVAILABLE_MB or grown > MAX_SWAP_GROWTH_MB:
                guard["tripped"] = f"MemAvailable {avail} MB, swap grown {grown} MB"
                print(f"MEMORY GUARD tripped in arm {name}: {guard['tripped']} - stopping the server", flush=True)
                stop(srv.pid)
                return
            time.sleep(1)

    import threading
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    t0 = time.time()
    while time.time() - t0 < 900 and not health(port).get("loaded"):
        if srv.poll() is not None:
            guard["done"] = True
            trace.stop()
            return {"arm": name, "error": "server died while loading", "memory_guard": guard["tripped"], "memtrace": trace.peaks()}
        time.sleep(5)
    if not health(port).get("loaded"):
        guard["done"] = True
        stop(srv.pid)
        trace.stop()
        return {"arm": name, "error": "not loaded in 15 minutes", "memory_guard": guard["tripped"], "memtrace": trace.peaks()}
    load_s = time.time() - t0
    p = subprocess.run([sys.executable, TPS, f"http://127.0.0.1:{port}", name, REPEATS, "256"], capture_output=True, text=True)
    guard["done"] = True
    watcher.join(timeout=30)
    if guard["tripped"] is None:
        stop(srv.pid)
    else:
        trace.stop()
        return {"arm": name, "error": "void: memory guard tripped", "memory_guard": guard["tripped"], "memtrace": trace.peaks()}
    trace.stop()
    oom_events = trace.peaks().get("events_at_end", {})
    with open(log, "rb") as f:
        f.seek(start_size)
        logtext = f.read().decode("utf-8", "replace")
    kinds, tokens = {}, 0
    for line in p.stdout.splitlines():
        if "decode tok/s:" in line:
            kind = line.split()[0]
            rates = [float(x) for x in line.split("decode tok/s:")[1].split("median")[0].replace(",", " ").split()]
            kinds[kind] = rates
            tokens += sum(int(x) for x in line.split("(tokens [")[1].split("]")[0].replace(",", " ").split())
    rows = io_lines(logtext)
    allr = sorted(x for v in kinds.values() for x in v)
    vm1 = vmstat()
    kernel = {k: vm1[k] - vm0[k] for k in vm1}
    kernel["swap_used_mb_before"] = swap_used0
    kernel["swap_used_mb_after"] = meminfo_mb("SwapTotal") - meminfo_mb("SwapFree")
    return {"arm": name, "env": env_extra, "engine_sha": engine_sha, "prefetch_on": "file tier io prefetch ON" in logtext,
            "memtrace": trace.peaks(), "kernel_during_arm": kernel, "files_dropped": dropped, "load_s": round(load_s), "decode_tok_s": kinds,
            "median_all": allr[len(allr) // 2] if allr else None, "tokens_timed": tokens,
            "requests_logged": len(rows), "os_read_mb": round(sum(r[0] for r in rows), 1), "expert_read_mb": round(sum(r[2] for r in rows), 1),
            "major_faults": sum(r[1] for r in rows),
            "os_mb_per_token": round(sum(r[0] for r in rows) / tokens, 1) if tokens else None, "stdout": p.stdout[-1500:]}


def main():
    cfg, log, out, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    results = []
    for arm in sys.argv[5:]:
        name, _, envs = arm.partition(":")
        env_extra = {} if envs in ("", "-") else dict(kv.split("=", 1) for kv in envs.split(","))
        print(f"=== arm {name} {env_extra}", flush=True)
        r = run_arm(name, env_extra, cfg, log, port)
        results.append(r)
        json.dump(results, open(out, "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "stdout"}), flush=True)
    print("DONE", flush=True)


main()
