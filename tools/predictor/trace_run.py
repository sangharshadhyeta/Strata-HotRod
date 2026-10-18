"""Router trace run: write a config for the trace engine (stand-ins off, STRATA_ROUTER_TRACE on, port 8097), start it in a memory-capped scope, drive it with
the prompts, report the trace file's coverage.  Stop with `systemctl stop <unit>.scope`.  Never touches 8095/8099 (the caller stops 8095 first)."""
import json, os, subprocess, sys, time, urllib.request

T = "/root/.claude/jobs/743230e4/tmp"
ROOT = "/home/Projects/Strata"
TRACE = "/mnt/nvme/strata-data/traces/trace1.bin"
PORT = 8097
cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
key = json.load(open(f"{T}/lan_server.info"))["api_key"]


def write_cfg():
    cfg = json.load(open(f"{ROOT}/strata-iq3_s.json"))
    cfg["exe"] = "/home/Projects/Strata-pred/build/strata"
    cfg["cwd"] = ROOT
    env = {k: v for k, v in cfg.get("env", {}).items() if not k.startswith("STRATA_ROUTE_RESIDENT")}     # exact routing: no stand-ins
    env["STRATA_ROUTER_TRACE"] = TRACE
    cfg["env"] = env
    a = cfg["args"]
    a[a.index("--max-context") + 1] = "32768"
    cfg.update({"host": "127.0.0.1", "port": PORT, "api_key": key, "log": f"{T}/trace_server.log"})
    json.dump(cfg, open(f"{T}/trace_cfg.json", "w"), indent=1)
    os.chmod(f"{T}/trace_cfg.json", 0o600)
    return cfg


def start():
    os.makedirs(os.path.dirname(TRACE), exist_ok=True)
    write_cfg()
    unit = f"strata-trace-{int(time.time())}"
    p = subprocess.Popen(["systemd-run", "--scope", "--quiet", f"--unit={unit}", "-p", "MemoryMax=29G", "-p", "MemorySwapMax=0",
                          f"{ROOT}/.venv/bin/python", f"{ROOT}/serve/server.py", "--engine", "strata", "--port", str(PORT), "--config", f"{T}/trace_cfg.json"],
                         cwd=ROOT, stdout=open(f"{T}/trace_server.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    open(f"{T}/trace.info", "w").write(json.dumps({"pid": p.pid, "unit": unit, "trace": TRACE}))
    print("started", p.pid, unit, flush=True)


def chat(prompt, max_tokens=450):
    body = json.dumps({"model": "strata", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["usage"]["completion_tokens"], time.time() - t0


def drive(limit=None):
    sys.path.insert(0, T)
    from trace_prompts import ALL
    total = 0
    for i, p in enumerate(ALL[:limit]):
        n, dt = chat(p)
        total += n
        print(f"[{i+1}/{len(ALL[:limit])}] {n} tokens in {dt:.0f} s ({n/dt:.1f} tok/s); total {total}; trace {os.path.getsize(TRACE)/1e9:.2f} GB", flush=True)
    print("DRIVER_DONE", total, flush=True)


if cmd == "start":
    start()
elif cmd == "drive":
    drive(int(sys.argv[2]) if len(sys.argv) > 2 else None)
