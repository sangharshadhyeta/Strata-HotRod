import sys, os, json, time, urllib.request
os.environ.setdefault("TRACE_ROOT", "/home/Projects/Strata-edge0")
sys.path.insert(0, "/home/Projects/Strata-edge0/tools/predictor")
import trace_prompts as TP
T = "/root/.claude/jobs/743230e4/tmp"; TR = "/mnt/nvme/strata-data/traces/edge0-trace.bin"
key = json.load(open(f"{T}/lan_server.info"))["api_key"]
done = 0
print("prompts", len(TP.ALL), flush=True)
START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for i, p in enumerate(TP.ALL):
    if i < START: continue
    if os.path.exists(TR) and os.path.getsize(TR) > 3.5e9:
        break
    body = json.dumps({"model": "q", "messages": [{"role": "user", "content": p}], "max_tokens": 450, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions", body, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=3000))
        done += 1
        print(f"prompt {i}: {r['usage']['completion_tokens']} tokens in {time.time()-t0:.0f} s, trace {os.path.getsize(TR)/1e9 if os.path.exists(TR) else 0:.2f} GB", flush=True)
    except Exception as e:
        print(f"prompt {i}: {e}", flush=True)
print("DONE", done, flush=True)
