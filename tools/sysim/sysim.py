"""sysim - a discrete-event trace of a decode window on a generic CPU + GPU + RAM + NVMe + PCIe machine.

Every expert blob, activation hand-off, kernel and copy of Strata's decode path is one task; tasks hold a server (the GPU stream, the
copy engine, the host thread, the CPU pool, an I/O thread) and move bytes over shared links (VRAM, PCIe, RAM, NVMe), which are split
max-min fairly among the transfers in flight.  From the trace it reports:
  * the CRITICAL PATH: the chain of tasks that sets the window time (what really blocks), by kind
  * WHO WAITED ON WHOM along it, and how much of each task's time was lost to sharing a link
  * how busy each resource was
  * SENSITIVITY: the speed change when one resource is made 1.5x faster (a "virtual speedup", as causal profilers do)
and writes a Chrome/Perfetto trace (open in ui.perfetto.dev or chrome://tracing) and a self-contained HTML timeline.

It is a model, calibrated to measured numbers, not the engine: no kernel runs.  Bit-level detail adds nothing a transfer-level model
misses - a 2.33 MB blob moves as one transfer at the share of the link it gets.

usage:
  python sysim.py [--profile rtx3060] [--windows 60] [--set key=value ...] [--html out.html] [--trace out.json] [--no-sens]
  python sysim.py --list                      (the profiles and every parameter)
"""
import argparse
import json
import math
import os
import random
import sys

# ---------------------------------------------------------------------------------------------------------------------------
# Profiles: hardware (bandwidths in bytes/s, latencies in s) + model + policy.  "rtx3060" is calibrated to docs/perf-ideas.md
# (RTX 3060 12 GB, i5-12400F 6 cores, DDR4, PCIe 3.0 NVMe 3.3 GB/s, Qwen3.8-Flash-Next IQ3_S): a ~55 ms window, ~25 ms of it the host
# waiting for the GPU, CPU experts ~10-15 ms, file reads ~8-13 ms.
PROFILES = {
    "rtx3060": dict(
        # hardware
        gpu_bw=360e9, gpu_eff=0.70, gpu_node_s=3.0e-6,       # VRAM bandwidth, the share kernels reach, cost of one graph node
        pcie_bw=12e9, pcie_lat=5e-6,                         # host<->GPU link (mapped rows, swap copies)
        ram_bw=38e9,                                         # DDR4 dual channel
        cpu_workers=6, cpu_core_bw=2.4e9, cpu_barrier_s=12e-6, cpu_phases=4,   # the expert pool: per-core expert-kernel rate
        memcpy_bw=9e9,                                       # one thread's memcpy (the O_DIRECT bounce copy)
        nvme_bw=3.3e9, nvme_lat=90e-6,                       # the drive (PCIe 3.0 x4 link) and one read's latency
        io_threads=8,                                        # STRATA_IO_PF_THREADS
        host_plan_s=40e-6, host_actq_s=15e-6, flag_lat=8e-6,
        # model (per decode window of `rows` tokens)
        layers=48, expert_bytes=2.33e6, rows=4.3, accepted=3.3,
        gpu_pre_bytes=34e6, gpu_pre_nodes=24,                # attention/GDN + hyper-connection read + router + look-ahead GEMVs
        gpu_post_bytes=12e6, gpu_post_nodes=10,              # combine + hyper-connection write
        vram_experts=27.0,                                   # distinct experts a layer computes from VRAM (hits + stand-ins)
        cpu_experts=1.5,                                     # distinct rank 0-1 misses a layer computes on the CPU
        cold_share=0.2,                                      # ... of them read from the drive (the rest are in the RAM tier)
        window_gpu_s=3.2e-3, window_host_s=0.6e-3,           # commit + MTP draft (GPU); sampling, token output (host)
        swaps=12.5, exchange_memcpy=0,                       # adaptive tier: swaps a window (H2D copies); 1 = host memcpy of the exchanges
        # read-ahead policy
        recall1=0.553, recall2=0.465, wrong_per_right=0.75,  # look-ahead: top-2 recall at dist 1 / 2; wrong guesses per right one
        edf=0, yield_=0,                                     # STRATA_IO_EDF, STRATA_IO_PF_YIELD
        zero_copy=0,                                         # 1: O_DIRECT reads land in the stage buffer (no bounce memcpy)
        # ideas from other fields (all off by default)
        conf_sep=0.25,      # how far apart the predictor's confidence is for right and wrong guesses (means 0.5 +- sep/2, sd 0.15)
        prefetch_theta=0.0, # Cao-Felten-Karlin-Li: read a guess only when its confidence clears this (0: every guess)
        value_order=0,      # read the guess with the most expected value first: confidence / layers to its deadline
        pace_s=0.0,         # Kingman: at least this long between two guess starts (smooths the bursts at each layer)
        recall3=0.0,        # multi-history (TAGE-like) predictor: guesses three layers ahead with this recall
        chunks=1,           # row-chunk streaming: a cold expert's rows are computed as each 1/chunks of it lands
        idle_fill=0,        # Theory of Constraints: RAM-tier promotions read ONLY while the drive is otherwise idle (1 thread)
        fill_useful=0.3,    # ... the share of promoted experts that a later window would have read cold (prevents that read)
        fill_chunk=2.33e6,  # ... read in pieces of this size, each started only while the drive is idle (smaller: steps aside sooner)
    ),
}
PROFILES["rtx4090-gen4"] = dict(PROFILES["rtx3060"], gpu_bw=1008e9, pcie_bw=24e9, ram_bw=70e9, cpu_workers=12, cpu_core_bw=3.5e9,
                                nvme_bw=7.0e9, nvme_lat=60e-6, vram_experts=29.0, cpu_experts=0.9, memcpy_bw=14e9)
PROFILES["rtx3060-model"] = dict(PROFILES["rtx3060"], recall1=0.69, wrong_per_right=0.75 * (1 - 0.69) / (1 - 0.553))   # STRATA_LA_MODEL

LINKS = ("vram", "pcie", "ram", "nvme")


# ---------------------------------------------------------------------------------------------------------------------------
class Task:
    __slots__ = ("id", "name", "cat", "server", "phases", "deps", "ndeps", "succ", "on_done", "ready", "start", "end", "pred",
                 "spred", "alone", "lane", "win", "layer", "pi", "rem", "rate", "res", "cap", "meta", "cancelled", "prio")

    def __init__(s, name, cat, server=None, phases=(), win=-1, layer=-1, prio=0.0, meta=None):
        s.name, s.cat, s.server, s.phases = name, cat, server, list(phases)
        s.deps, s.ndeps, s.succ, s.on_done = [], 0, [], []
        s.ready = s.start = s.end = None
        s.pred = s.spred = None
        s.alone = 0.0
        s.lane = None
        s.win, s.layer, s.prio, s.meta = win, layer, prio, meta or {}
        s.pi, s.rem, s.rate, s.res, s.cap = -1, 0.0, 0.0, (), math.inf
        s.cancelled = False


class Sim:
    def __init__(s, p):
        s.p = p
        s.t = 0.0
        s.bw = {"vram": p["gpu_bw"], "pcie": p["pcie_bw"], "ram": p["ram_bw"], "nvme": p["nvme_bw"]}
        s.servers = {"gpu": [1, 0, []], "copy": [1, 0, []], "host": [1, 0, []], "cpu": [1, 0, []], "io": [p["io_threads"], 0, []],
                     "fill": [1, 0, []]}
        s.last_release = {k: None for k in s.servers}
        s.lanes_free = {k: list(range(v[0])) for k, v in s.servers.items()}
        s.running = []
        s.tasks = []
        s.busy_bytes = {k: 0.0 for k in LINKS}
        s.server_busy = {k: 0.0 for k in s.servers}
        s.pick = {}          # server -> function(queue) -> index
        s.parent = None      # the task whose completion is running callbacks now (it caused what they add)
        s.gate = {}          # server -> function() -> bool (may a queued task start now)
        s.nid = 0

    # -- building
    def add(s, task, deps=()):
        task.id = s.nid
        s.nid += 1
        s.tasks.append(task)
        for d in deps:
            if d is None:
                continue
            task.deps.append(d)
            if d.end is None:
                task.ndeps += 1
                d.succ.append(task)
        a = 0.0
        for ph in task.phases:
            if ph[0] == "lat":
                a += ph[1]
            else:
                _, nbytes, res, cap = ph
                a += nbytes / min([cap] + [s.bw[r] for r in res])
        task.alone = a
        if task.ndeps == 0:
            done = [d for d in task.deps if d is not None and d.end is not None]
            task.pred = max(done, key=lambda d: d.end) if done else s.parent
            if task.pred is not None and s.parent is not None and s.parent.end is not None and s.parent.end > task.pred.end:
                task.pred = s.parent
            s._ready(task)
        return task

    def _ready(s, task):
        task.ready = s.t
        if task.server is None:
            s._start(task)
        else:
            s.servers[task.server][2].append(task)

    def _start(s, task):
        task.start = s.t
        if task.server is not None:
            srv = s.servers[task.server]
            srv[1] += 1
            task.lane = s.lanes_free[task.server].pop(0)
            if task.start > task.ready + 1e-12:
                task.spred = s.last_release[task.server]
        task.pi = -1
        s._next_phase(task)

    def _next_phase(s, task):
        task.pi += 1
        if task.pi >= len(task.phases):
            s._finish(task)
            return
        ph = task.phases[task.pi]
        if ph[0] == "lat":
            task.rem, task.res, task.cap = ph[1], (), math.inf
        else:
            task.rem, task.res, task.cap = ph[1], ph[2], ph[3]
        if task.rem <= 0:
            s._next_phase(task)
            return
        s.running.append(task)

    def _finish(s, task):
        task.end = s.t
        if task.server is not None:
            srv = s.servers[task.server]
            srv[1] -= 1
            s.lanes_free[task.server].append(task.lane)
            s.lanes_free[task.server].sort()
            s.last_release[task.server] = task
            s.server_busy[task.server] += task.end - task.start
        s.parent = task
        for f in task.on_done:
            f(task)
        s.parent = None
        for n in task.succ:
            n.ndeps -= 1
            if n.ndeps == 0 and not n.cancelled:
                n.pred = task                 # the dependency that finished last made it ready
                s._ready(n)

    def cancel(s, task):
        """A queued task that will not run (a guess the decode read itself)."""
        task.cancelled = True
        q = s.servers[task.server][2] if task.server else []
        if task in q:
            q.remove(task)

    # -- running
    def _schedule(s):
        moved = True
        while moved:
            moved = False
            for name, srv in s.servers.items():
                while srv[2] and srv[1] < srv[0]:
                    g = s.gate.get(name)
                    if g is not None and not g():
                        break
                    i = s.pick[name](srv[2]) if name in s.pick else min(range(len(srv[2])), key=lambda j: (srv[2][j].prio, srv[2][j].ready, srv[2][j].id))
                    if i is None:
                        break
                    task = srv[2].pop(i)
                    if task.cancelled:
                        continue
                    s._start(task)
                    moved = True

    def _rates(s):
        flows = [t for t in s.running if t.res]
        for t in s.running:
            t.rate = 0.0
        rem = dict(s.bw)
        live = set(flows)
        while live:
            n = {r: 0 for r in LINKS}
            for f in live:
                for r in f.res:
                    n[r] += 1
            inc = min([rem[r] / n[r] for r in LINKS if n[r]] + [f.cap - f.rate for f in live])
            inc = max(inc, 0.0)
            for f in live:
                f.rate += inc
                for r in f.res:
                    rem[r] -= inc
            frozen = {f for f in live if f.cap - f.rate <= 1e-9 * max(1.0, f.cap if f.cap < math.inf else 1.0)
                      or any(rem[r] <= 1e-6 * s.bw[r] for r in f.res)}
            if not frozen:
                break
            live -= frozen

    def run(s, until_done):
        while True:
            s._schedule()
            if until_done():
                return
            if not s.running:
                raise RuntimeError(f"deadlock at t={s.t:.6f}: nothing runs; queues " +
                                   str({k: [t.name for t in v[2]] for k, v in s.servers.items() if v[2]}))
            s._rates()
            dt = min((t.rem if not t.res else t.rem / t.rate) for t in s.running)
            for t in s.running:
                if t.res:
                    moved = t.rate * dt
                    t.rem -= moved
                    for r in t.res:
                        s.busy_bytes[r] += moved
                else:
                    t.rem -= dt
            s.t += dt
            done = [t for t in s.running if (t.rem <= (1.0 if t.res else 1e-12))]
            s.running = [t for t in s.running if t not in done]
            for t in done:
                s._next_phase(t)


# ---------------------------------------------------------------------------------------------------------------------------
def pois(rng, lam):
    if lam <= 0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p < L:
            return k
        k += 1


def build_and_run(p, windows, seed):
    """Strata's decode path, window by window.  Returns the sim and the windows' (start, end) times."""
    rng = random.Random(seed)
    sim = Sim(p)
    gpu_rate = p["gpu_bw"] * p["gpu_eff"]
    L = int(p["layers"])
    state = {"cur": (-1, -1), "demand_active": 0}
    guesses = {}        # (win, layer, expert) -> guess task

    # the I/O threads' order: newest guess first (today), or next layer first (STRATA_IO_EDF)
    def io_pick(q):
        if not q:
            return None
        if p["edf"]:
            w, l = state["cur"]
            live = [i for i, t in enumerate(q) if t.win == w and t.layer > l]
            if not live:
                for t in q:
                    t.cancelled = True
                q.clear()
                return None
            if p["value_order"]:
                return min(live, key=lambda i: (q[i].layer, -q[i].meta["conf"], q[i].id))
            return min(live, key=lambda i: (q[i].layer, -q[i].meta["from"], q[i].id))
        if p["value_order"]:
            l = state["cur"][1]
            return max(range(len(q)), key=lambda i: q[i].meta["conf"] / max(1, q[i].layer - l))
        return len(q) - 1
    sim.pick["io"] = io_pick
    sim.gate["io"] = lambda: not (p["yield_"] and state["demand_active"] > 0)
    # the idle filler starts a promotion only when no other read is on the drive and none waits to start
    sim.gate["fill"] = lambda: (not any("nvme" in t.res for t in sim.running) and not sim.servers["io"][2]
                                and state["demand_active"] == 0)
    state["credit"] = 0.0

    def fill_done(t):
        state["credit"] += p["fill_useful"]

    def gpu_task(name, cat, nbytes, nodes, w, l, deps):
        ph = [("lat", nodes * p["gpu_node_s"]), ("flow", nbytes, ("vram",), gpu_rate)]
        return sim.add(Task(name, cat, "gpu", ph, w, l), deps)

    def read_phases(bounce):
        ph = [("lat", p["nvme_lat"]), ("flow", p["expert_bytes"], ("nvme", "ram"), math.inf)]
        if bounce and not p["zero_copy"]:
            ph.append(("flow", p["expert_bytes"], ("ram",), p["memcpy_bw"]))
        return ph

    marks = []
    prev_post = None          # the previous layer's GPU post (or the previous window's draft)
    prev_host = None
    for w in range(windows):
        w_start = sim.t if prev_post is None else None
        # this window's misses: cold (from the drive) and warm (RAM tier) CPU experts, per layer
        cold = [[(w, l, e) for e in rng.sample(range(512), min(512, pois(rng, p["cpu_experts"] * p["cold_share"])))] for l in range(L)]
        if p["idle_fill"]:   # this window's candidates; each piece starts only while the drive is otherwise idle
            n_piece = max(1, int(math.ceil(p["expert_bytes"] / p["fill_chunk"])))
            for i in range(24):
                prev = prev_post
                for j in range(n_piece):
                    f = Task(f"w{w} promotion {i}" + (f" piece {j}" if n_piece > 1 else ""), "NVMe: idle-time promotion", "fill",
                             ([("lat", p["nvme_lat"])] if j == 0 else []) + [("flow", p["expert_bytes"] / n_piece, ("nvme", "ram"), math.inf)], w, -1)
                    sim.add(f, [prev] if prev is not None else [])
                    prev = f
                prev.on_done.append(fill_done)
        warm = [pois(rng, p["cpu_experts"] * (1 - p["cold_share"])) for l in range(L)]
        pred1 = [[k for k in cold[l] if rng.random() < p["recall1"]] for l in range(L)]
        pred2 = [[k for k in pred1[l] if rng.random() < p["recall2"] / max(p["recall1"], 1e-9)] for l in range(L)]
        pred3 = [[k for k in cold[l] if rng.random() < p["recall3"]] for l in range(L)]
        win_tasks = {}
        if p["exchange_memcpy"] and w > 0:   # the adaptive tier's exchanges copied on the host before the window
            prev_host = sim.add(Task("exchange memcpy", "host: exchange memcpy", "host",
                                     [("flow", p["swaps"] * p["expert_bytes"], ("ram",), p["memcpy_bw"])], w, -1), [prev_host])
        for l in range(L):
            pre = gpu_task(f"w{w} L{l} attention+router", "GPU: attention/router", p["gpu_pre_bytes"], p["gpu_pre_nodes"], w, l, [prev_post])
            if w_start is None and l == 0:
                w_start = pre
            plan = sim.add(Task(f"w{w} L{l} plan", "host: plan", "host", [("lat", p["flag_lat"] + p["host_plan_s"])], w, l), [pre, prev_host])

            def on_plan(task, w=w, l=l):
                state["cur"] = (w, l)
                # the look-ahead's guesses for the next two layers, read ahead on the I/O threads
                for dist, preds in ((1, pred1), (2, pred2), (3, pred3)):
                    T = l + dist
                    if T >= L or not p[f"recall{dist}"]:
                        continue
                    right = list(preds[T])
                    nwrong = pois(rng, max(1, len(cold[T])) * p["wrong_per_right"] * p[f"recall{dist}"])
                    keys = [(k, True) for k in right] + [((w, T, 1000 + rng.randrange(10 ** 9)), False) for _ in range(nwrong)]
                    rng.shuffle(keys)
                    for k, ok in keys:
                        if k in guesses:
                            continue
                        conf = min(1.0, max(0.0, rng.gauss(0.5 + (p["conf_sep"] / 2 if ok else -p["conf_sep"] / 2), 0.15)))
                        if conf < p["prefetch_theta"]:
                            continue                       # not worth the drive time it takes from the others
                        g = Task(f"w{w} guess L{T}", "I/O: read-ahead", "io", read_phases(True), w, T, meta={"from": l, "conf": conf})
                        guesses[k] = g
                        deps = []
                        if p["pace_s"] > 0:                # one guess start per pace_s at most: a timer in front of it
                            slot = max(sim.t, state.get("pace_next", 0.0))
                            state["pace_next"] = slot + p["pace_s"]
                            if slot > sim.t:
                                deps.append(sim.add(Task("pace", "sync", None, [("lat", slot - sim.t)], w, T)))
                        sim.add(g, deps)
            plan.on_done.append(on_plan)
            vram = gpu_task(f"w{w} L{l} VRAM experts", "GPU: VRAM experts", p["vram_experts"] * p["expert_bytes"], 6, w, l, [plan])
            actq = sim.add(Task(f"w{w} L{l} act quant", "host: plan", "host", [("lat", p["host_actq_s"])], w, l), [plan])
            # the demand reads: decided when the host reaches them (what the read-ahead has done by then)
            holder = {}

            def demand(task, w=w, l=l, holder=holder):
                waits, reads = [], []
                for k in cold[l]:
                    g = guesses.get(k)
                    if g is not None and g.end is not None:
                        continue                                   # landed in time
                    if g is not None and g.start is not None:
                        waits.append(g)                            # still being read: wait for it
                        continue
                    if g is not None:
                        sim.cancel(g)                              # queued: the host reads it now
                    if p["idle_fill"] and state["credit"] >= 1.0:  # an idle-time promotion put it in the RAM tier
                        state["credit"] -= 1.0
                        continue
                    reads.append(k)
                n = len(reads)
                C = max(1, int(p["chunks"]))
                holder["waits"], holder["n"] = waits, n
                holder["bounce"] = []
                if not n:
                    return
                state["demand_active"] += 1
                eb = p["expert_bytes"] / C
                prev = [None] * n
                for j in range(C):                 # chunk j of every demand read (one read when chunks = 1), then its copy
                    rd = []
                    for i in range(n):
                        ph = ([("lat", p["nvme_lat"])] if j == 0 else []) + [("flow", eb, ("nvme", "ram"), math.inf)]
                        prev[i] = sim.add(Task(f"w{w} L{l} demand read" + (f" chunk {j}" if C > 1 else ""), "NVMe: demand read", None, ph, w, l),
                                          [prev[i]] if prev[i] is not None else [])
                        rd.append(prev[i])
                    b = sim.add(Task(f"w{w} L{l} bounce copy", "host: O_DIRECT copy", "host",
                                     [("flow", 0 if p["zero_copy"] else n * eb, ("ram",), p["memcpy_bw"])], w, l), rd)
                    holder["bounce"].append(b)
                holder["bounce"][-1].on_done.append(lambda t: state.__setitem__("demand_active", state["demand_active"] - 1))
            actq.on_done.append(demand)
            nexp = len(cold[l]) + warm[l]
            cpu_bytes = nexp * p["expert_bytes"]
            gate = sim.add(Task(f"w{w} L{l} reads done", "sync", None, [], w, l), [actq])
            cpu = Task(f"w{w} L{l} CPU experts", "CPU: experts", "cpu",
                       [("lat", p["cpu_barrier_s"] * p["cpu_phases"] if nexp else 0.0),
                        ("flow", cpu_bytes, ("ram",), p["cpu_workers"] * p["cpu_core_bw"])], w, l)
            win_tasks[l] = (gate, cpu, holder)

            # the gate's own deps come from demand(): add the CPU task once the host has decided (actq done)
            def add_cpu(task, gate=gate, cpu=cpu, holder=holder, nexp=nexp):
                waits, n, bounce = holder.get("waits", []), holder.get("n", 0), holder.get("bounce", [])
                rate = p["cpu_workers"] * p["cpu_core_bw"]
                if int(p["chunks"]) <= 1 or n == 0:
                    sim.add(cpu, [gate] + waits + bounce)
                    return
                # row-chunk streaming: the experts that are here first, then each chunk of the demand reads as it lands
                C = len(bounce)
                ready = sim.add(Task(cpu.name + " (ready ones)", "CPU: experts", "cpu",
                                     [("lat", p["cpu_barrier_s"] * p["cpu_phases"]),
                                      ("flow", (nexp - n) * p["expert_bytes"], ("ram",), rate)], cpu.win, cpu.layer), [gate] + waits)
                prev = ready
                for j in range(C - 1):
                    prev = sim.add(Task(cpu.name + f" chunk {j}", "CPU: experts", "cpu",
                                        [("lat", p["cpu_barrier_s"]), ("flow", n * p["expert_bytes"] / C, ("ram",), rate)], cpu.win, cpu.layer),
                                   [prev, bounce[j]])
                cpu.phases = [("lat", p["cpu_barrier_s"]), ("flow", n * p["expert_bytes"] / C, ("ram",), rate)]
                sim.add(cpu, [prev, bounce[-1]])
            gate.on_done.append(add_cpu)
            # the GPU waits for the CPU's rows (flag over PCIe), then combines; the host goes on to the next layer only after its
            # pool has run (expert_pool_dispatch_multi returns after the CPU work)
            rows = sim.add(Task(f"w{w} L{l} rows to GPU", "PCIe: rows+flag", None,
                                [("lat", p["pcie_lat"]), ("flow", max(1, nexp) * p["rows"] * 2560 * 4, ("pcie",), math.inf)], w, l), [])
            rows.ndeps += 1
            cpu.succ.append(rows)
            cpu.on_done.append(lambda t: None)
            post = gpu_task(f"w{w} L{l} combine", "GPU: combine/HC write", p["gpu_post_bytes"], p["gpu_post_nodes"], w, l, [vram])
            post.ndeps += 1
            rows.succ.append(post)
            prev_post, prev_host = post, cpu
        draft = sim.add(Task(f"w{w} commit+draft", "GPU: commit+MTP draft", "gpu", [("lat", p["window_gpu_s"])], w, L), [prev_post])
        hostw = sim.add(Task(f"w{w} sample+emit", "host: window", "host", [("lat", p["window_host_s"])], w, L), [draft, prev_host])
        for i in range(int(round(p["swaps"]))):   # adaptive tier swaps: H2D copies beside the next window
            sim.add(Task(f"w{w} swap {i}", "PCIe: tier swaps", "copy", [("flow", p["expert_bytes"], ("pcie", "ram"), math.inf)], w, L),
                    [hostw])
        marks.append((w_start, hostw))
        prev_post, prev_host = hostw, hostw
    sim.run(lambda: all(m[1].end is not None for m in marks))
    return sim, marks


# ---------------------------------------------------------------------------------------------------------------------------
def analyse(sim, marks, p, skip=3):
    """Critical path (backwards from the last window's end), who waited on whom, sharing losses, utilisation."""
    first = marks[skip][0].start if len(marks) > skip else 0.0
    last = marks[-1][1]
    T = last.end - first
    nwin = len(marks) - skip
    crit, edges, share = {}, {}, {}
    cur = last
    path = []
    while cur is not None and cur.end is not None and cur.end > first:
        s0 = max(cur.start, first)
        crit[cur.cat] = crit.get(cur.cat, 0.0) + (cur.end - s0)
        lost = max(0.0, (cur.end - cur.start) - cur.alone)
        if lost > 0:
            share[cur.cat] = share.get(cur.cat, 0.0) + min(lost, cur.end - s0)
        path.append(cur)
        if cur.start > cur.ready + 1e-12 and cur.spred is not None:
            nxt, why = cur.spred, "queued behind"
            gap = cur.start - cur.ready
            crit["(queue) " + cur.server] = crit.get("(queue) " + cur.server, 0.0)   # time is counted in the holder's own entry
        else:
            nxt, why = cur.pred, "waited for"
        if nxt is not None:
            key = (cur.cat, why, nxt.cat)
            edges[key] = edges.get(key, 0.0) + (cur.end - s0)
        cur = nxt
    crit = {k: v for k, v in crit.items() if v > 0}
    util = {k: sim.busy_bytes[k] / (sim.bw[k] * sim.t) for k in LINKS}
    for k, v in sim.server_busy.items():
        util["server " + k] = v / (sim.t * sim.servers[k][0])
    win_ms = T / nwin * 1e3
    return dict(T=T, nwin=nwin, win_ms=win_ms, tok_s=p["accepted"] / (win_ms / 1e3), crit=crit, edges=edges, share=share, util=util,
                path=set(id(t) for t in path))


SENS = [("NVMe bandwidth", "nvme_bw"), ("RAM bandwidth", "ram_bw"), ("CPU core speed", "cpu_core_bw"), ("GPU memory bandwidth", "gpu_bw"),
        ("PCIe bandwidth", "pcie_bw"), ("one-thread memcpy", "memcpy_bw"), ("graph node overhead (1/1.5)", "gpu_node_s"),
        ("host plan time (1/1.5)", "host_plan_s"), ("look-ahead recall", "recall1")]


def sensitivity(p, windows, seed, base_tok):
    out = []
    for label, key in SENS:
        q = dict(p)
        if key.endswith("_s"):
            q[key] = p[key] / 1.5
        elif key == "recall1":
            q["recall1"] = min(1.0, p["recall1"] * 1.5)
            q["recall2"] = min(1.0, p["recall2"] * 1.5)
        else:
            q[key] = p[key] * 1.5
        sim, marks = build_and_run(q, windows, seed)
        r = analyse(sim, marks, q)
        out.append((label, (r["tok_s"] / base_tok - 1) * 100))
    return sorted(out, key=lambda x: -x[1])


# ---------------------------------------------------------------------------------------------------------------------------
def tok(p, windows, seeds):
    v = []
    for sd in seeds:
        sim, marks = build_and_run(p, windows, sd)
        v.append(analyse(sim, marks, p)["tok_s"])
    return sum(v) / len(v)


def ideal(p):
    """The ideal machine for this model: every expert already on the GPU, kernels at the full VRAM bandwidth, no graph-node or
    host hand-off cost.  What is left is the bytes the GPU must stream per window (and the draft)."""
    q = dict(p)
    q.update(gpu_eff=1.0, gpu_node_s=0.0, host_plan_s=0.0, host_actq_s=0.0, flag_lat=0.0, pcie_lat=0.0, cpu_barrier_s=0.0,
             vram_experts=p["vram_experts"] + p["cpu_experts"], cpu_experts=0.0, swaps=0.0, window_host_s=0.0, exchange_memcpy=0)
    return q


WATERFALL = [
    ("kernels below the VRAM bandwidth (gpu_eff)", dict(gpu_eff=None)),
    ("graph-node launch cost", dict(gpu_node_s=None)),
    ("host hand-off each layer (doorbell, plan, quantise, flags)", dict(host_plan_s=None, host_actq_s=None, flag_lat=None, pcie_lat=None, window_host_s=None)),
    ("rank 0-1 misses computed on the CPU (from RAM)", dict(vram_experts=None, cpu_experts=None, cpu_barrier_s=None, cold_share=0.0)),
    ("tier swaps over PCIe", dict(swaps=None)),
    ("cold misses read from the drive, perfect read-ahead", dict(cold_share=None, recall1=1.0, recall2=1.0, wrong_per_right=0.0)),
    ("the read-ahead misses guesses (recall)", dict(recall1=None, recall2=None, wrong_per_right=0.0)),
    ("wrong guesses share the drive", dict(wrong_per_right=None)),
]
FIXES = [
    ("STRATA_IO_PF_THREADS=1 (existing)", dict(io_threads=1)),
    ("+ STRATA_IO_PF_YIELD=1", dict(yield_=1)),
    ("+ STRATA_IO_EDF=1", dict(edf=1)),
    ("+ STRATA_LA_MODEL (trained predictor)", dict(recall1=0.69, wrong_per_right=0.75 * (1 - 0.69) / (1 - 0.553))),
    ("+ zero-copy O_DIRECT reads (not built)", dict(zero_copy=1)),
    # GPU side, from the kernel review (estimates: measure with STRATA_DECODE_STAGE_TIMERS first)
    ("+ router/look-ahead fused, HC norm folded, hit-add fused (-12 nodes a layer)", dict(gpu_pre_nodes=14, gpu_post_nodes=8)),
    ("+ GPU kernels 0.70 -> 0.80 of VRAM bandwidth (MMVF_ROWS, GDN_SPLIT, EXPDEC...)", dict(gpu_eff=0.80)),
]


def waterfall(p, windows, seeds):
    q = ideal(p)
    rows = [("IDEAL: every expert on the GPU, no overheads", tok(q, windows, seeds))]
    for label, ch in WATERFALL:
        for k, v in ch.items():
            q[k] = p[k] if v is None else v
        rows.append((label, tok(q, windows, seeds)))
    real = rows[-1][1]
    fixes = []
    for label, ch in FIXES:
        q.update(ch)
        fixes.append((label, tok(q, windows, seeds)))
    return rows, fixes, q


# ---------------------------------------------------------------------------------------------------------------------------
def write_trace(sim, marks, path, res, wins=4, skip=3):
    lo, hi = marks[skip][0].start, marks[min(len(marks) - 1, skip + wins - 1)][1].end
    ev = []
    pid = {"gpu": 1, "copy": 2, "host": 3, "cpu": 4, "io": 5, None: 6}
    names = {1: "GPU stream", 2: "GPU copy engine", 3: "host thread", 4: "CPU expert pool", 5: "I/O read-ahead threads", 6: "NVMe demand reads / links", 7: "critical path"}
    for k, v in names.items():
        ev.append({"ph": "M", "name": "process_name", "pid": k, "args": {"name": v}})
    for t in sim.tasks:
        if t.start is None or t.end is None or t.end < lo or t.start > hi or t.cat == "sync":
            continue
        d = {"name": t.name, "cat": t.cat, "ph": "X", "ts": (t.start - lo) * 1e6, "dur": max(0.01, (t.end - t.start) * 1e6),
             "pid": pid[t.server], "tid": t.lane or 0, "args": {"kind": t.cat, "alone_us": round(t.alone * 1e6, 1),
                                                                "waited_for_server_us": round((t.start - t.ready) * 1e6, 1)}}
        ev.append(d)
        if id(t) in res["path"]:
            ev.append(dict(d, pid=7, tid=0))
    with open(path, "w") as f:
        json.dump({"traceEvents": ev, "displayTimeUnit": "ms"}, f)


def write_html(sim, marks, path, res, sens, p, prof, wins=3, skip=3):
    lo, hi = marks[skip][0].start, marks[min(len(marks) - 1, skip + wins - 1)][1].end
    lanes = []
    def lane_of(t):
        if t.server == "gpu":
            return "GPU stream"
        if t.server == "copy":
            return "GPU copy engine"
        if t.server == "host":
            return "host thread"
        if t.server == "cpu":
            return "CPU expert pool"
        if t.server == "io":
            return f"I/O thread {t.lane}"
        if t.cat.startswith("NVMe"):
            return "NVMe demand reads"
        return "PCIe rows/flags"
    rows = []
    for t in sim.tasks:
        if t.start is None or t.end is None or t.end < lo or t.start > hi or t.cat == "sync":
            continue
        rows.append([lane_of(t), round((t.start - lo) * 1e3, 4), round((t.end - lo) * 1e3, 4), t.cat, t.name,
                     1 if id(t) in res["path"] else 0, round((t.start - t.ready) * 1e3, 4)])
    order = ["GPU stream", "GPU copy engine", "host thread", "CPU expert pool", "NVMe demand reads", "PCIe rows/flags"] + \
            [f"I/O thread {i}" for i in range(p["io_threads"])]
    data = dict(rows=rows, order=order, span=round((hi - lo) * 1e3, 3), crit=sorted(res["crit"].items(), key=lambda x: -x[1]),
                total=res["T"], share=res["share"], util=res["util"], sens=sens, tok=res["tok_s"], win=res["win_ms"], prof=prof,
                edges=sorted([[a, why, b, v] for (a, why, b), v in res["edges"].items()], key=lambda x: -x[3])[:12])
    html = HTML.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decode system trace</title><style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e3e0d8;--card:#fff;--crit:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe7;--mut:#a3a19b;--line:#33322e;--card:#1f1f1c;--crit:#fb923c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.kpi b{display:block;font-size:22px}
canvas{width:100%;display:block}#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:6px 8px;font-size:12px;display:none;max-width:320px;box-shadow:0 2px 8px #0002}
.bar{display:grid;grid-template-columns:240px 1fr 70px;gap:8px;align-items:center;margin:3px 0;font-size:13px}
.bar i{display:block;height:12px;border-radius:3px}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}
.legend span{display:inline-flex;align-items:center;gap:4px;margin:0 10px 4px 0;font-size:12px}.legend i{width:10px;height:10px;border-radius:2px}
@media (max-width:640px){.bar{grid-template-columns:130px 1fr 56px}}
</style></head><body><main>
<h1>Decode system trace</h1><div class="mut" id="sub"></div>
<div class="kpis" id="kpis"></div>
<h2>What sets the window time: the critical path</h2><div class="card"><div class="mut" style="margin-bottom:6px">Every task on the chain that ends the window, by kind. A faster resource only helps if its work is on this chain.</div><div id="crit"></div></div>
<h2>Timeline (3 decode windows; outlined = on the critical path)</h2><div class="card"><div class="legend" id="legend"></div><canvas id="tl"></canvas><div class="mut">Drag to zoom a range, double-click to reset. Hover a block for details.</div></div>
<h2>Who waited on whom (along the critical path)</h2><div class="card"><table id="edges"></table></div>
<h2>If one resource were 1.5x faster</h2><div class="card"><div id="sens"></div></div>
<h2>How busy each resource was</h2><div class="card"><div id="util"></div></div>
<div id="tip"></div></main><script>
const D=__DATA__;
const pal={"GPU: attention/router":"#4f7cac","GPU: VRAM experts":"#6aa3d8","GPU: combine/HC write":"#93c0e8","GPU: commit+MTP draft":"#2f5d8a",
"host: plan":"#8e7cc3","host: O_DIRECT copy":"#b4a7e0","host: window":"#6b5b95","host: exchange memcpy":"#c9b8f0","CPU: experts":"#d68a3c",
"NVMe: demand read":"#c2410c","I/O: read-ahead":"#e8a87c","PCIe: rows+flag":"#7fb07f","PCIe: tier swaps":"#5c9a5c"};
const col=c=>pal[c]||"#999";const $=id=>document.getElementById(id);
$("sub").textContent=`profile ${D.prof} - a model calibrated to measurements, not an engine run`;
const tk=[["window",D.win.toFixed(1)+" ms"],["tokens/s (model)",D.tok.toFixed(1)],["critical path, top",D.crit[0][0]],["its share",(100*D.crit[0][1]/D.total).toFixed(0)+"%"]];
$("kpis").innerHTML=tk.map(([a,b])=>`<div class="card kpi"><span class="mut">${a}</span><b>${b}</b></div>`).join("");
const tot=D.total;$("crit").innerHTML=D.crit.map(([k,v])=>`<div class="bar"><span>${k}</span><i style="width:${100*v/D.crit[0][1]}%;background:${col(k)}"></i><span class="mut">${(100*v/tot).toFixed(1)}%</span></div>`).join("")
+Object.entries(D.share).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([k,v])=>`<div class="mut" style="font-size:12px">${k}: ${(100*v/tot).toFixed(1)}% of the window lost to sharing a link</div>`).join("");
$("edges").innerHTML="<tr><th>task</th><th></th><th>on</th><th class=n>share of window</th></tr>"+D.edges.map(([a,w,b,v])=>`<tr><td>${a}</td><td class=mut>${w}</td><td>${b}</td><td class=n>${(100*v/tot).toFixed(1)}%</td></tr>`).join("");
const mx=Math.max(...D.sens.map(s=>Math.abs(s[1])),1);
$("sens").innerHTML=D.sens.map(([k,v])=>`<div class="bar"><span>${k}</span><i style="width:${100*Math.abs(v)/mx}%;background:${v>=0?'#4f9d69':'#c0504d'}"></i><span class="mut">${v>=0?'+':''}${v.toFixed(1)}%</span></div>`).join("");
$("util").innerHTML=Object.entries(D.util).map(([k,v])=>`<div class="bar"><span>${k}</span><i style="width:${Math.min(100,100*v)}%;background:#8a8a8a"></i><span class="mut">${(100*v).toFixed(0)}%</span></div>`).join("");
$("legend").innerHTML=Object.keys(pal).filter(k=>D.rows.some(r=>r[3]==k)).map(k=>`<span><i style="background:${pal[k]}"></i>${k}</span>`).join("");
const cv=$("tl"),ctx=cv.getContext("2d"),lanes=D.order.filter(l=>D.rows.some(r=>r[0]==l)),LH=22,LW=120;let v0=0,v1=D.span,drag=null;
function draw(){const dpr=devicePixelRatio||1,W=cv.clientWidth,H=lanes.length*LH+24;cv.width=W*dpr;cv.height=H*dpr;cv.style.height=H+"px";ctx.setTransform(dpr,0,0,dpr,0,0);
const cs=getComputedStyle(document.documentElement),fg=cs.getPropertyValue("--fg"),mut=cs.getPropertyValue("--mut"),ln=cs.getPropertyValue("--line"),cr=cs.getPropertyValue("--crit");
ctx.clearRect(0,0,W,H);ctx.font="11px system-ui";const x=t=>LW+(t-v0)/(v1-v0)*(W-LW-4);
lanes.forEach((l,i)=>{ctx.fillStyle=mut;ctx.fillText(l,4,i*LH+15);ctx.fillStyle=ln;ctx.fillRect(LW,i*LH+LH-1,W-LW,1)});
for(const r of D.rows){const i=lanes.indexOf(r[0]);if(i<0||r[2]<v0||r[1]>v1)continue;const a=x(r[1]),b=Math.max(x(r[2]),a+1);ctx.fillStyle=col(r[3]);ctx.fillRect(a,i*LH+4,b-a,LH-9);
if(r[5]){ctx.strokeStyle=cr;ctx.lineWidth=1.5;ctx.strokeRect(a,i*LH+3,b-a,LH-7)}}
const step=Math.pow(10,Math.floor(Math.log10((v1-v0)/5)));ctx.fillStyle=mut;for(let t=Math.ceil(v0/step)*step;t<=v1;t+=step){const a=x(t);ctx.fillRect(a,lanes.length*LH,1,4);ctx.fillText(t.toFixed(step<1?1:0)+" ms",a+2,lanes.length*LH+16)}}
function hit(e){const R=cv.getBoundingClientRect(),W=cv.clientWidth,px=e.clientX-R.left,py=e.clientY-R.top,i=Math.floor(py/LH),t=v0+(px-LW)/(W-LW-4)*(v1-v0);
return D.rows.find(r=>r[0]==lanes[i]&&r[1]<=t&&r[2]>=t)}
cv.addEventListener("mousemove",e=>{const r=hit(e),tip=$("tip");if(!r||drag){tip.style.display="none";return}tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
tip.innerHTML=`<b>${r[4]}</b><br>${r[3]}<br>${(r[2]-r[1]).toFixed(3)} ms${r[6]>0.0005?`, queued ${r[6].toFixed(3)} ms first`:""}${r[5]?"<br><b>on the critical path</b>":""}`});
cv.addEventListener("mouseleave",()=>$("tip").style.display="none");
const tAt=e=>{const R=cv.getBoundingClientRect();return v0+(e.clientX-R.left-LW)/(cv.clientWidth-LW-4)*(v1-v0)};
cv.addEventListener("mousedown",e=>drag=tAt(e));cv.addEventListener("mouseup",e=>{if(drag===null)return;const t=tAt(e);if(Math.abs(t-drag)>(v1-v0)/200){v0=Math.max(0,Math.min(t,drag));v1=Math.min(D.span,Math.max(t,drag))}drag=null;draw()});
cv.addEventListener("dblclick",()=>{v0=0;v1=D.span;draw()});addEventListener("resize",draw);matchMedia("(prefers-color-scheme:dark)").addEventListener("change",draw);draw();
</script></body></html>"""


# ---------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (see --list)")
    ap.add_argument("--html")
    ap.add_argument("--trace")
    ap.add_argument("--no-sens", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--waterfall", action="store_true", help="ideal -> real, one constraint at a time; then the fixes")
    a = ap.parse_args()
    if a.list:
        for k, v in PROFILES.items():
            print(k, json.dumps(v, indent=1))
        return
    p = dict(PROFILES[a.profile])
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in p:
            sys.exit(f"unknown parameter {k}")
        p[k] = type(p[k])(float(v)) if not isinstance(p[k], int) else int(float(v))
    if a.waterfall:
        seeds = [a.seed, a.seed + 1, a.seed + 2]
        rows, fixes, fixed = waterfall(p, a.windows, seeds)
        print(f"profile {a.profile}: from the ideal machine to today, one real constraint at a time (tokens/s, model, mean of 3 seeds)\n")
        prev = rows[0][1]
        print(f"  {rows[0][0]:62s} {prev:6.1f}")
        for label, v in rows[1:]:
            print(f"  + {label:60s} {v:6.1f}  {(v - prev) / rows[0][1] * 100:+6.1f}% of ideal")
            prev = v
        print(f"\n  TODAY {rows[-1][1]:.1f} tok/s = {rows[-1][1] / rows[0][1] * 100:.0f}% of the ideal {rows[0][1]:.1f}\n\nfixes, applied one after another:")
        prev = rows[-1][1]
        for label, v in fixes:
            print(f"  {label:62s} {v:6.1f}  {(v / rows[-1][1] - 1) * 100:+6.1f}% vs today")
        print(f"\n  after the fixes: {fixes[-1][1] / rows[0][1] * 100:.0f}% of the ideal")
        p = fixed
    sim, marks = build_and_run(p, a.windows, a.seed)
    r = analyse(sim, marks, p)
    print(f"profile {a.profile}: window {r['win_ms']:.1f} ms, {r['tok_s']:.1f} tok/s (model; {p['accepted']} tokens a window), {r['nwin']} windows\n")
    print("critical path (what sets the window time):")
    for k, v in sorted(r["crit"].items(), key=lambda x: -x[1]):
        print(f"  {k:28s} {v / r['T'] * 100:5.1f}%  {v / r['nwin'] * 1e3:6.2f} ms a window")
    print("\nlost to sharing a link, on the critical path:")
    for k, v in sorted(r["share"].items(), key=lambda x: -x[1])[:6]:
        print(f"  {k:28s} {v / r['nwin'] * 1e3:6.2f} ms a window")
    print("\nwho waited on whom (on the critical path):")
    for (x, why, y), v in sorted(r["edges"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {x:28s} {why:13s} {y:28s} {v / r['T'] * 100:5.1f}%")
    print("\nutilisation:")
    for k, v in r["util"].items():
        print(f"  {k:28s} {v * 100:5.1f}%")
    sens = []
    if not a.no_sens:
        sens = sensitivity(p, a.windows, a.seed, r["tok_s"])
        print("\nif one resource were 1.5x faster:")
        for label, d in sens:
            print(f"  {label:32s} {d:+6.1f}%")
    if a.trace:
        write_trace(sim, marks, a.trace, r)
        print(f"\ntrace: {a.trace} (open in ui.perfetto.dev or chrome://tracing)")
    if a.html:
        write_html(sim, marks, a.html, r, sens, p, a.profile)
        print(f"timeline: {a.html}")


if __name__ == "__main__":
    main()
