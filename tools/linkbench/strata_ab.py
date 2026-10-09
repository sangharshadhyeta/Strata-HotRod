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


def run_arm(name, env_extra, cfg, log, port):
    dropped = drop_cache()
    start_size = os.path.getsize(log) if os.path.exists(log) else 0
    env = dict(os.environ)
    env.update(env_extra)
    srv = subprocess.Popen([ROOT + "/.venv/bin/python", ROOT + "/serve/server.py", "--engine", "strata", "--config", cfg, "--port", str(port)],
                           cwd=ROOT, env=env, stdout=open("/root/.claude/jobs/743230e4/tmp/strata_ab_server.out", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900 and not health(port).get("loaded"):
        if srv.poll() is not None:
            return {"arm": name, "error": "server died while loading"}
        time.sleep(5)
    if not health(port).get("loaded"):
        stop(srv.pid)
        return {"arm": name, "error": "not loaded in 15 minutes"}
    load_s = time.time() - t0
    p = subprocess.run([sys.executable, TPS, f"http://127.0.0.1:{port}", name, "3", "256"], capture_output=True, text=True)
    stop(srv.pid)
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
    return {"arm": name, "env": env_extra, "files_dropped": dropped, "load_s": round(load_s), "decode_tok_s": kinds,
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
