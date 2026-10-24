"""Router trace run: write a config for the trace engine (stand-ins off, STRATA_ROUTER_TRACE on, and with SET=prefill STRATA_ROUTER_TRACE_PREFILL=1 so the
prompts' own routing is traced too), start it in a memory-capped scope, drive it with the prompts, report the trace file's size.  Stop with
`systemctl stop <unit>.scope`.  Never touches 8095, 8097 or 8099 (the caller stops whatever runs there).

Environment (all optional):
  TRACE            trace file (default /mnt/nvme/strata-data/traces/trace-prefill.bin)
  SET              prompt set: "prefill" (conversations of long prompts, trace_prompts.PREFILL) or "all" (trace_prompts.ALL, the decode set)
  CFG              engine config to copy (default /home/Projects/Strata/strata-iq3_s-abl.json)
  TRACE_MAX_GB     stop driving when the trace file is larger than this (default 12)
  PORT             server port (default 8111)
  EXE              engine binary (default /home/Projects/Strata-train/build/strata)
  TRACE_PYTHON     python for serve/server.py (default /home/Projects/Strata/.venv/bin/python)

Usage: trace_run.py start | drive [limit]      (drive with SET=prefill runs every conversation; limit applies to SET=all)
"""
import json, os, subprocess, sys, time, urllib.request

T = "/root/.claude/jobs/743230e4/tmp"
ROOT = "/home/Projects/Strata-train"
TRACE = os.environ.get("TRACE", "/mnt/nvme/strata-data/traces/trace-prefill.bin")
SET = os.environ.get("SET", "prefill")
CFG = os.environ.get("CFG", "/home/Projects/Strata/strata-iq3_s-abl.json")
TRACE_MAX_GB = float(os.environ.get("TRACE_MAX_GB", "12"))
PORT = int(os.environ.get("PORT", "8111"))
EXE = os.environ.get("EXE", "/home/Projects/Strata-train/build/strata")
PY = os.environ.get("TRACE_PYTHON", "/home/Projects/Strata/.venv/bin/python")
FORBIDDEN_PORTS = (8095, 8097, 8099)
assert PORT not in FORBIDDEN_PORTS, f"port {PORT} is reserved: pick another PORT"
assert SET in ("prefill", "all"), SET
cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
key = json.load(open(f"{T}/lan_server.info"))["api_key"] if os.path.exists(f"{T}/lan_server.info") else ""


def write_cfg():
    cfg = json.load(open(CFG))
    cfg["exe"] = EXE
    cfg["cwd"] = ROOT
    env = {k: v for k, v in cfg.get("env", {}).items() if not k.startswith("STRATA_ROUTE_RESIDENT")}     # exact routing: no stand-ins
    env["STRATA_ROUTER_TRACE"] = TRACE
    if SET == "prefill":
        env["STRATA_ROUTER_TRACE_PREFILL"] = "1"
        if os.environ.get("STRATA_ROUTER_TRACE_PREFILL_EVERY"):
            env["STRATA_ROUTER_TRACE_PREFILL_EVERY"] = os.environ["STRATA_ROUTER_TRACE_PREFILL_EVERY"]
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
                          PY, f"{ROOT}/serve/server.py", "--engine", "strata", "--port", str(PORT), "--config", f"{T}/trace_cfg.json"],
                         cwd=ROOT, stdout=open(f"{T}/trace_server.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    open(f"{T}/trace.info", "w").write(json.dumps({"pid": p.pid, "unit": unit, "trace": TRACE, "port": PORT}))
    print("started", p.pid, unit, "port", PORT, "trace", TRACE, flush=True)


def chat(messages, max_tokens=400):
    """One chat request over the whole message list.  Returns (completion tokens, seconds, the reply text)."""
    body = json.dumps({"model": "strata", "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["usage"]["completion_tokens"], time.time() - t0, r["choices"][0]["message"]["content"]


def over_cap():
    size = os.path.getsize(TRACE) if os.path.exists(TRACE) else 0
    if size > TRACE_MAX_GB * 1e9:
        print(f"STOP: trace {size / 1e9:.2f} GB is over TRACE_MAX_GB={TRACE_MAX_GB}", flush=True)
        return True
    return False


def drive(limit=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trace_prompts
    total = requests = 0
    if SET == "prefill":
        conversations = trace_prompts.PREFILL
        n_req = sum(len(c) for c in conversations)
        for ci, conv in enumerate(conversations):
            messages = []
            for ti, (topic, text) in enumerate(conv):
                if over_cap():
                    print("DRIVER_STOPPED", total, requests, flush=True)
                    return
                messages.append({"role": "user", "content": text})
                n, dt, reply = chat(messages)
                messages.append({"role": "assistant", "content": reply})
                total += n
                requests += 1
                print(f"[{requests}/{n_req}] conv {ci + 1} turn {ti + 1} ({topic}, {len(text)} chars): {n} tokens in {dt:.0f} s; "
                      f"total {total}; trace {os.path.getsize(TRACE) / 1e9:.2f} GB", flush=True)
    else:
        prompts = trace_prompts.ALL[:limit]
        for i, p in enumerate(prompts):
            if over_cap():
                print("DRIVER_STOPPED", total, i, flush=True)
                return
            n, dt, _ = chat([{"role": "user", "content": p}])
            total += n
            requests += 1
            print(f"[{i + 1}/{len(prompts)}] {n} tokens in {dt:.0f} s ({n / dt:.1f} tok/s); total {total}; "
                  f"trace {os.path.getsize(TRACE) / 1e9:.2f} GB", flush=True)
    print("DRIVER_DONE", total, requests, flush=True)


if cmd == "start":
    start()
elif cmd == "drive":
    drive(int(sys.argv[2]) if len(sys.argv) > 2 else None)
