"""Live view of memtrace_F1c.csv: one line per 15 s, an ALERT line at once when swap or memory pressure or an oom event appears."""
import csv, os, sys, time

path = sys.argv[1]
result = sys.argv[2]
last_t, last_change, prev_fault, alerted, every = -1.0, time.time(), 0, set(), 15
next_line = 0.0
while True:
    rows = []
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        pass
    if rows:
        r = rows[-1]
        t = float(r["t_s"])
        if t != last_t:
            last_t, last_change = t, time.time()
            alerts = []
            if float(r["sys_swap_used"]) > 200: alerts.append(f"swap in use {r['sys_swap_used']} MB")
            if float(r["sys_avail"]) < 3000: alerts.append(f"MemAvailable {r['sys_avail']} MB")
            if int(r["ev_max"]) > 0 or int(r["ev_oom"]) > 0 or int(r["ev_oom_kill"]) > 0:
                alerts.append(f"cgroup events max={r['ev_max']} oom={r['ev_oom']} oom_kill={r['ev_oom_kill']}")
            for a in alerts:
                if a.split()[0] not in alerted or "oom" in a:
                    alerted.add(a.split()[0])
                    print(f"ALERT t={t:.0f}s: {a}", flush=True)
            if t >= next_line:
                dfault = int(r["cg_pgmajfault"]) - prev_fault
                prev_fault = int(r["cg_pgmajfault"])
                print(f"t={t:5.0f}s scope {float(r['cg_current'])/1024:5.1f} GB (anon {float(r['cg_anon'])/1024:4.1f}, file {float(r['cg_file'])/1024:4.1f}) "
                      f"engine rss {float(r['eng_rss'])/1024:4.1f} GB locked {float(r['eng_locked'])/1024:4.1f} | system avail {float(r['sys_avail'])/1024:4.1f} GB "
                      f"swap {r['sys_swap_used']} MB | high={r['ev_high']} max={r['ev_max']} oom_kill={r['ev_oom_kill']} | major faults +{dfault}", flush=True)
                next_line = t + every
    if os.path.exists(result) or (rows and time.time() - last_change > 20):
        print("TRACE ENDED", flush=True)
        break
    time.sleep(2)
