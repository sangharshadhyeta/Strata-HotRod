"""Tune the engine's hardware-dependent settings on this PC (setup's --calibrate).

Three settings depend on the machine more than on the model, and the defaults are right for the PC they were
measured on (a Ryzen 5 7600 + RTX 5070 on PCIe 5):
  --pcie-frac     the share of the experts missing from VRAM that are copied over PCIe and run on the GPU instead of
                  on the CPU.  A fast PCIe link and a slow CPU want more; a laptop's x8 link or a fast CPU want less.
  --spec-min-p    how sure the draft layer must be to extend a verify window by another guess.  A slower CPU pays more
                  per extra window row (more experts per window), so it wants a higher floor.
  --pool-workers  the CPU threads that compute experts.  Every physical core is not always best: on hybrid CPUs the
                  efficiency cores can make the whole window wait for them.
  --adapt-every / --adapt-swaps / --adapt-decay
                  how the VRAM expert tier follows the conversation: how often, how many experts it swaps in, how fast
                  its use counts fade.  A PC whose CPU reads the missed experts slowly (DDR3, ~24 GB/s) gains from
                  swapping more: 160 swaps every window measured +7.9% over the default tier on a Xeon E5-2673 v3 (DDR3) with an
                  RTX 4060 Ti on PCIe 3.0 x8 (Q2_0; with the swaps taking effect a window later, #764).
  --prefill-pipe  a layer split only: the prompt chunks that keep the stages reading at once.  The best chunk count
                  depends on b/a, the fixed cost of a chunk over that of a token, which differs by an order of
                  magnitude between rigs (about 700 tokens on four GP100 dies, 5000 on two RTX 3090 Ti) and costs
                  when it is wrong (a fixed 1024 read 22-36% slower on two RTX 4090) - so it is measured here, once,
                  as the prompt read speed of fresh prompts, rather than guessed or learned while the PC serves.
  STRATA_SH_STREAM (AMD only)
                  the shared expert on its own stream.  Off by default on HIP (#816: an RX 6800 on Windows and 4x R9700
                  decode 12-45% faster without it), but the same card family on another PC goes the other way (#1272:
                  2x RX 6900 XT on Linux, 6-11% faster with it), so it is measured here, per PC, and written into the
                  config's "env" only when it wins.  The output is identical either way (same token ids).

  our features (HotRod, 21 Nov)
                  the learners are trained here, and each output-changing or per-window feature is measured against its off
                  state: the learned look-ahead (STRATA_LA_LEARN), stand-ins (STRATA_ROUTE_RESIDENT), the route tail skip
                  (STRATA_ROUTE_TAIL_SKIP) and the mask (STRATA_MASK).  Every one is ON by default, because it won on the PC
                  it was built on (a 12 GB RTX 3060); a PC with a bigger card misses fewer experts, and a feature that costs a
                  fixed time per window then loses.  A feature is turned OFF here only when off is more than MIN_GAIN faster,
                  and the choice is written into the config's "env".  The learners start from nothing and train in this run
                  (no downloaded state); they keep learning live afterwards.

The first two and the prompt chunks are measured through one engine (per-request `strata_tune` keys); the worker
count, the adaptive tier and the shared-expert stream need a restart per value.  Those are decode speed: the prompt
path streams every expert whatever they say.

A setting is kept only when it beats the default by more than MIN_GAIN in an interleaved re-measurement - the
adaptive expert tier and the OS make single measurements noisy by a few percent.

    python tools/calibrate.py strata-q2_0.json        # measure and print; setup.py --calibrate also saves it
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

MIN_GAIN = 0.03                    # a setting must beat the default by this much to be kept
PCIE_FRACS = (0.0, 0.2, 0.35, 0.55, 0.75, 0.9, 1.0)   # 1.0: every miss over PCIe, the CPU pool gets no expert
AUTO = -1.0                        # the PCIe share the engine chooses itself (PcieAuto, STRATA_PCIE_BALANCE): no per-request value is sent
SPEC_MIN_PS = (0.3, 0.5, 0.7)
SWEEP_ROUNDS = 3                   # how many times the sweep visits each value (see `sweep`)
# the adaptive tier's candidates (every, swaps, decay) against the engine's own (None): swapping more and remembering
# longer, the rest of the set as the engine has it
# HIP: the shared-expert stream fork, off (the HIP default) against on; an engine environment variable, not an argument
SH_STREAM = "STRATA_SH_STREAM"
ADAPT_CANDIDATES = (None, ("1", "80", "0.97"), ("1", "160", "0.97"))
ADAPT_FLAGS = ("--adapt-every", "--adapt-swaps", "--adapt-decay")
MAX_NEW = 128
# the expert tier follows a text over some windows: 128-token answers end before it shows (on a Xeon E5-2673 v3 with
# DDR3, every 1 / 160 / 0.97 measured +0.7% with 128 tokens and +7.9% with 512-token answers), so its step uses these
TIER_MAX_NEW = 512
# --prefill-pipe's values: 0 the buffers' chunks (the engine default), 1 their count evened out, then b/a in tokens
PIPE_VALUES = (0, 1, 768, 1536, 3072, 6144)
PIPE_TOKENS = (2048, 8192)        # fresh prompts read per value (their text: the install's docs/*.md)
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List twelve European capitals with one sentence about each.",
)


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)


def arg_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    """`args` with `flag value` set (replaced if present), or removed when value is None."""
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def worker_candidates(default: int, extra=()) -> list[int]:
    """The engine's own count, and fewer: two thirds, a half and a quarter (at least 2), plus `extra` (the P-cores
    minus one on a hybrid CPU, one socket's cores minus one on a 2-socket PC), without repeats.  Bench #780 #815
    #707 #674: 4 beat 15 by 28.7% on 8P + 16E; 17-18 beat 35 by ~20% on 2-socket Xeons."""
    c = [default]
    for w in (round(default * 2 / 3), round(default / 2), round(default / 4), *extra):
        if 2 <= w < default and w not in c:
            c.append(w)
    return c


def host_worker_extras() -> list[int]:
    """P-cores - 1 (hybrid CPU) and one socket's cores - 1 (several sockets) of this PC: more counts to try."""
    try:
        import setup
        out = []
        cores, sockets = setup.cpu_cores(), setup.cpu_sockets()
        if cores:
            out.append(cores[0] - 1)
        if sockets and sockets[0] >= 2:
            out.append(sockets[1] - 1)
        return out
    except Exception:
        return []


def doc_prompt(tok):
    """`make(i, n)`: a fresh prompt of about n tokens - the same stretch of the docs' text for every i, after a header
    that differs for every i, so no read reuses an earlier one's cache and the reads differ only in their chunks (where
    experts are streamed, the read speed depends on the text: a different stretch per read measured the text)."""
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted((ROOT / "docs").glob("*.md")))
    body = tok.encode(text, parse_special=False) or [0]
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)

    def make(i: int, n: int) -> list[int]:
        head = tok.encode(f"<|im_start|>user\n[calibration read {i}] Summarize:\n", parse_special=True)
        return head + (body * (n // len(body) + 1))[:n] + tail
    return make


def measure_pipe(eng, make, say=print) -> tuple[float, dict]:
    """--prefill-pipe on one running engine (a layer split): the prompt read speed of fresh prompts per value, then the
    best against the default (0) in SWEEP_ROUNDS paired rounds (back to back, the order alternating).  It is kept only
    when it is faster in every round and by more than MIN_GAIN in the median: on two RTX 3090 Ti one read can be 20%
    off, and a b/a too small for the rig reads 19-24% slower there.  One pass over the values, not `sweep`'s rounds:
    every value reads the same text (`doc_prompt`), and on four GP100 dies one pass already takes about 4 minutes."""
    reads = [0]

    def rate(v: float) -> float:
        ms = toks = 0.0
        for n in PIPE_TOKENS:
            reads[0] += 1
            sampling = {"temperature": 0, "strata_checkpoint": False, "strata_tune": {"prefill_pipe": v}}
            for _ in eng.generate(make(reads[0], n), 1, sampling, threading.Event()):
                pass
            last = eng.last or {}
            ms += float(last.get("prompt_ms") or 0.0)
            toks += float(last.get("prompt_read") or (last.get("prompt_tokens") or 0) - (last.get("reused") or 0))
        return toks / (ms / 1000.0) if ms > 0 and toks > 0 else 0.0

    rate(0)                                            # the first long read after a start faults the weights in
    by_value = {}
    for v in PIPE_VALUES:
        by_value[v] = [rate(v)]
        say(f"    prompt chunks {'off' if v == 0 else 'evened out' if v == 1 else f'b/a {v}'}: {by_value[v][0]:.0f} tok/s")
    best = max(by_value, key=lambda k: by_value[k][0])
    confirm = {0: [], best: []}
    chosen = 0
    if best != 0:
        for r in range(SWEEP_ROUNDS):
            for k in ((0, best) if r % 2 == 0 else (best, 0)):
                confirm[k].append(rate(k))
        gains = [c / d - 1.0 for d, c in zip(confirm[0], confirm[best]) if d > 0]
        if len(gains) == SWEEP_ROUNDS and min(gains) > 0.0 and statistics.median(gains) > MIN_GAIN:
            chosen = best
    return chosen, {"pipe_sweep": {str(k): v for k, v in by_value.items()},
                    "pipe_confirm": {str(k): v for k, v in confirm.items()}}


def tuned_default_tier(args: list[str]) -> list[str]:
    """`args` with the adaptive tier's flags removed: the engine's own tier."""
    out = list(args)
    for flag in ADAPT_FLAGS:
        out = with_arg(out, flag, None)
    return out


def ptune(frac: float) -> dict:
    """The per-request tune keys for a PCIe share: none for the engine's own (AUTO), else the fixed value."""
    return {} if frac == AUTO else {"pcie_frac": frac}


def pick(measured: dict, default_key, min_gain: float = MIN_GAIN):
    """The key with the best median tok/s, or `default_key` unless the best beats it by more than min_gain."""
    med = {k: statistics.median(v) for k, v in measured.items() if v}
    if not med or default_key not in med:
        return default_key
    best = max(med, key=med.get)
    return best if med[best] > med[default_key] * (1.0 + min_gain) else default_key


class Session:
    """One running engine: measure decode tok/s for a setting (the median of the prompts' rates)."""

    def __init__(self, engine, ids_list):
        self.engine = engine
        self.ids_list = ids_list

    def rate(self, tune: dict | None = None, max_new: int | None = None) -> float:
        rates = []
        for ids in self.ids_list:
            sampling = {"temperature": 0}
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, max_new or MAX_NEW, sampling, threading.Event())
                    if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
        return statistics.median(rates) if rates else 0.0

    def warm_up(self, rounds: int = 2):
        for _ in range(rounds):
            self.rate()


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {flag: value}, "report": {...}}.
    `start_engine(args)` returns a started engine (serve.server.StrataEngine or a stand-in in tests)."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env

        def start_engine(args, env=None):
            e = child_env(cfg)
            e.update(env or {})
            return StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=e)
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    ids_list = [chat_ids(tok, p) for p in PROMPTS]
    hip = cfg.get("backend") == "hip"
    cfg_env = cfg.get("env") or {}
    features = () if hip else tuple(f for f in FEATURES if f[0] in cfg_env or f[0] in FEATURES_WITHOUT_ENV)
    return measure(engine_args(cfg), ids_list, start_engine, say, host_worker_extras(), doc_prompt(tok),
                   hip=hip, features=features)


def engine_args(cfg: dict) -> list[str]:
    """The arguments the server starts this config's engine with (serve.server.engine_args), so the tuning measures
    the engine as it runs.  #447: this used to read any "gpu" list as a layer split and add --layer-split auto, which
    broke a one-card config with a helper card for the expert tier ("gpu": [0] + --expert-cache-device1: the engine
    refuses a split that leaves no card without a stage), missed the "0,2" spelling, and ignored split_skip_if_fits."""
    from serve.server import engine_args as server_args
    return server_args(cfg)


def engine_error(log: str | None, since: int = 0) -> str | None:
    """#447: the engine's own reason for a failed start - the last line of its log written after byte `since` that is
    its own ("strata ..." or "ERR ..."), else the last line there; None without a log or a new line."""
    if not log:
        return None
    try:
        with open(log, "rb") as f:
            f.seek(since)
            lines = [x.strip() for x in f.read()[-16384:].decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return None
    return next((x for x in reversed(lines) if x.startswith(("strata", "ERR"))), lines[-1] if lines else None)


def sweep(s, values: list, base_tune: dict, key: str, label: str, say) -> dict:
    """Every value, `SWEEP_ROUNDS` times, one round visiting all of them and every other round in reverse order.

    A single rate per value cannot order neighbours.  #1332 has 0.75 to 0.95 inside each other's noise at 72.7,
    69.0, 71.0 and 71.9, and that box moved 8-28% between engine starts at an unchanged setting; a 5080/Xeon box
    reports +9.57 tok/s between two blocks at an unchanged setting (p = 0.00004), larger than the effect being
    hunted.  Rotating the values inside one engine start puts that drift on every value the same number of times,
    and the median over the rounds keeps a single bad sample from deciding.  The caller takes the winner from the
    medians, so every value keeps the same chance the others had.
    """
    out = {v: [] for v in values}
    for r in range(SWEEP_ROUNDS):
        for v in (values if r % 2 == 0 else values[::-1]):
            out[v].append(s.rate({**base_tune, **({} if v == AUTO else {key: v})}))
        say(f"    round {r + 1}/{SWEEP_ROUNDS} {label}: " +
            "  ".join(f"{'auto' if v == AUTO else format(v, 'g')} {out[v][-1]:.1f}" for v in values))
    med = {v: statistics.median(out[v]) for v in values}
    say(f"    {label}, median of {SWEEP_ROUNDS}: " +
        ", ".join(f"{'auto' if v == AUTO else format(v, 'g')}: {med[v]:.1f} ({min(out[v]):.1f}-{max(out[v]):.1f})" for v in values) +
        f" -> best {'auto' if max(med, key=lambda k: med[k]) == AUTO else format(max(med, key=lambda k: med[k]), 'g')}")
    return out


def measure(base_args: list[str], ids_list, start_engine, say=print, extra_workers=(), long_prompt=None, hip: bool = False,
            features=()) -> dict:
    t0 = time.time()
    report: dict = {}
    say("  Loading the model for the measurements ...")
    base_args = apply(base_args, {})                   # the product defaults: what the measurements must beat
    eng = start_engine(base_args)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        # 21 Nov: when the engine's share is its own (pcie_auto=1: PcieAuto or STRATA_PCIE_BALANCE, no --pcie-frac in the args) the default is
        # that, not the number it prints: a fixed share set per request replaces the balance, so measuring 0.55 would measure something the
        # engine does not run
        auto_pcie = bool(int(info.get("pcie_auto", 0) or 0)) and arg_value(base_args, "--pcie-frac") is None
        d_pcie = AUTO if auto_pcie else float(info.get("pcie_frac", 0.55))
        d_minp = float(info.get("spec_min_p", 0.5))
        d_workers = int(info.get("pool_workers", 0)) or None
        s = Session(eng, ids_list)
        s.warm_up()
        # 0. a layer split's prompt chunks (prompt read speed; independent of the decode settings below)
        pipe = 0
        # the engine's own count of prompt stages where it reports one (a split can leave one, e.g.
        # --split-skip-if-fits), else the flag
        stages = info.get("prefill_stages")
        split = stages > 1 if isinstance(stages, int) else "--layer-split" in base_args
        if long_prompt is not None and split:
            say("  Measuring the prompt chunks of the layer split ...")
            pipe, rep_pipe = measure_pipe(eng, long_prompt, say)
            report.update(rep_pipe)
        # 1. the PCIe share, at the default draft floor
        by_pcie = sweep(s, sorted(set(PCIE_FRACS) | {AUTO if auto_pcie else round(d_pcie, 2)}), {"spec_min_p": d_minp}, "pcie_frac",
                        "PCIe share", say)
        best_pcie = max(by_pcie, key=lambda k: statistics.median(by_pcie[k]))
        # 2. the draft floor, at that share
        by_minp = sweep(s, sorted(set(SPEC_MIN_PS) | {round(d_minp, 2)}), ptune(best_pcie), "spec_min_p",
                        "draft floor", say)
        best_minp = max(by_minp, key=lambda k: statistics.median(by_minp[k]))
        # 3. the winner against the default, interleaved, the same number of rounds
        dflt, cand = (d_pcie if auto_pcie else round(d_pcie, 2), round(d_minp, 2)), (best_pcie, best_minp)
        confirm = {dflt: [], cand: []}
        if cand != dflt:
            for _ in range(SWEEP_ROUNDS):
                for k in (dflt, cand):
                    confirm[k].append(s.rate({**ptune(k[0]), "spec_min_p": k[1]}))
        chosen = pick(confirm, dflt) if cand != dflt else dflt
        report.update(default={"pcie_frac": dflt[0], "spec_min_p": dflt[1], "pool_workers": d_workers},
                      pcie_sweep={str(k): v for k, v in by_pcie.items()},
                      min_p_sweep={str(k): v for k, v in by_minp.items()},
                      confirm={f"{k[0]}/{k[1]}": v for k, v in confirm.items()})
    finally:
        close(eng)
    settings = {}
    if pipe:
        settings["--prefill-pipe"] = str(int(pipe))
    if chosen != dflt:
        if chosen[0] != AUTO:                          # a fixed share only when it beat the engine's own by more than MIN_GAIN
            settings["--pcie-frac"] = f"{chosen[0]:.2f}"
        settings["--spec-min-p"] = f"{chosen[1]:.2f}"
    base_rate = statistics.median(confirm[chosen]) if confirm.get(chosen) else None
    # 4. fewer CPU workers (a restart each), with the chosen settings
    if d_workers and len(worker_candidates(d_workers, extra_workers)) > 1:
        tuned = with_arg(with_arg(base_args, "--pcie-frac", None if chosen[0] == AUTO else f"{chosen[0]:.2f}"), "--spec-min-p",
                         f"{chosen[1]:.2f}")
        by_workers = {}
        for w in worker_candidates(d_workers, extra_workers):
            say(f"  Measuring with {w} CPU workers (restarts the engine) ...")
            e = restart(start_engine, with_arg(tuned, "--pool-workers", None if w == d_workers else str(w)), f"{w} workers",
                        say, report)
            if e is None:
                continue
            try:
                sw = Session(e, ids_list)
                sw.warm_up(1)
                by_workers[w] = [sw.rate(), sw.rate()]
                say(f"    {w} workers: {statistics.median(by_workers[w]):.1f} tok/s")
            finally:
                close(e)
        if len(by_workers) > 1 and d_workers in by_workers:   # the default again, last: the arms are bracketed (a warm-up or drift over the series
            say(f"  Measuring with {d_workers} CPU workers again (restarts the engine) ...")   # lands on the first arm, not on the candidates)
            e = restart(start_engine, with_arg(tuned, "--pool-workers", None), f"{d_workers} workers again", say, report)
            if e is not None:
                try:
                    sw = Session(e, ids_list)
                    sw.warm_up(1)
                    by_workers[d_workers] += [sw.rate(), sw.rate()]
                    say(f"    {d_workers} workers, again: {statistics.median(by_workers[d_workers][-2:]):.1f} tok/s")
                finally:
                    close(e)
        report["workers"] = {str(k): v for k, v in by_workers.items()}
        w_best = pick(by_workers, d_workers) if by_workers else d_workers
        if w_best != d_workers:
            settings["--pool-workers"] = str(w_best)
            base_rate = statistics.median(by_workers[w_best])
        elif by_workers.get(d_workers):
            base_rate = statistics.median(by_workers[d_workers])
    # 5. the adaptive expert tier (a restart each), with everything chosen so far
    tuned = apply(base_args, settings)
    by_adapt = {}
    for cand in ADAPT_CANDIDATES:
        key = "default" if cand is None else "/".join(cand)
        args = tuned
        for flag, v in zip(ADAPT_FLAGS, cand or (None,) * len(ADAPT_FLAGS)):
            args = with_arg(args, flag, v)
        say(f"  Measuring the expert tier {'(the engine default)' if cand is None else 'every ' + cand[0] + ', ' + cand[1] + ' swaps, decay ' + cand[2]} (restarts the engine) ...")
        e = restart(start_engine, args, "the expert tier " + key, say, report)
        if e is None:
            continue
        try:
            sa = Session(e, ids_list)
            sa.warm_up(1)
            by_adapt[key] = [sa.rate(max_new=TIER_MAX_NEW), sa.rate(max_new=TIER_MAX_NEW)]
            say(f"    {statistics.median(by_adapt[key]):.1f} tok/s")
        finally:
            close(e)
    if len(by_adapt) > 1 and "default" in by_adapt:   # the engine's own tier again, last (see the worker counts above)
        say("  Measuring the expert tier (the engine default) again (restarts the engine) ...")
        e = restart(start_engine, tuned_default_tier(tuned), "the expert tier default again", say, report)
        if e is not None:
            try:
                sa = Session(e, ids_list)
                sa.warm_up(1)
                by_adapt["default"] += [sa.rate(max_new=TIER_MAX_NEW), sa.rate(max_new=TIER_MAX_NEW)]
                say(f"    again: {statistics.median(by_adapt['default'][-2:]):.1f} tok/s")
            finally:
                close(e)
    a_best = pick(by_adapt, "default") if by_adapt else "default"
    report["adapt"] = by_adapt
    if a_best != "default":
        for flag, v in zip(ADAPT_FLAGS, a_best.split("/")):
            settings[flag] = v
        base_rate = statistics.median(by_adapt[a_best])
    elif by_adapt.get("default"):
        base_rate = statistics.median(by_adapt["default"])
    # 5b. our features (a restart each), with everything chosen so far.  The learners train first (a feature that learns must be
    # measured trained: an untrained learner is all cost), then each feature is measured against its off state, greedily: a
    # feature turned off stays off for the ones after it.
    env = {}
    if features:
        tuned = apply(base_args, settings)
        if any(f[0] == "STRATA_LA_LEARN" for f in features):
            say("  Training the learners on this PC (restarts the engine) ...")
            e = restart(lambda a: start_engine(a, {"STRATA_LA_LEARN": "1"}), tuned, "the learners' training", say, report)
            if e is not None:
                try:
                    st = Session(e, ids_list)
                    st.warm_up(1)
                    for _ in range(SEED_PROMPTS // max(1, len(ids_list)) + 1):
                        st.rate(max_new=TIER_MAX_NEW)
                finally:
                    close(e)                               # QUIT: the engine writes what it learned (data/predictor-<id>.bin ...)
        rep_f = {}
        for key, off, label in features:
            if key == "STRATA_LA_LEARN":                   # on = learn (what the training above left); off = 0
                on_env = {"STRATA_LA_LEARN": "1"}
            else:
                on_env = {}
            by_f = {}
            # on, off, on: the first arm of a restart series pays a warm-up (the page cache, the tier) that a single "on" then "off" would
            # hand to "off" (the tail skip read 22.5 on / 25.9 off in a first real run and tied in two alternated A/Bs); the default is
            # measured at both ends and its median is taken over both
            for state, extra in (("on", on_env), ("off", {key: off}), ("on", on_env)):
                say(f"  Measuring {label} {state} (restarts the engine) ...")
                e = restart(lambda a, x=extra: start_engine(a, {**env, **x}), tuned, f"{label} {state}", say, report)
                if e is None:
                    continue
                try:
                    sf = Session(e, ids_list)
                    sf.warm_up(1)
                    by_f.setdefault(state, []).extend([sf.rate(max_new=TIER_MAX_NEW), sf.rate(max_new=TIER_MAX_NEW)])
                    say(f"    {label} {state}: {statistics.median(by_f[state][-2:]):.1f} tok/s")
                finally:
                    close(e)
            rep_f[key] = by_f
            if pick(by_f, "on") == "off":
                env[key] = off
                say(f"    {label}: more than {MIN_GAIN:.0%} faster off on this PC - turned off")
                base_rate = statistics.median(by_f["off"])
            elif by_f.get("on"):
                base_rate = statistics.median(by_f["on"])
        # the opt-in switches: off (the default) against on, on kept only when more than MIN_GAIN faster
        for key, on, label in OPT_IN_SWITCHES:
            by_f = {}
            for state, extra in (("off", {}), ("on", {key: on}), ("off", {})):   # the default at both ends
                say(f"  Measuring {label} {state} (restarts the engine) ...")
                e = restart(lambda a, x=extra: start_engine(a, {**env, **x}), tuned, f"{label} {state}", say, report)
                if e is None:
                    continue
                try:
                    sf = Session(e, ids_list)
                    sf.warm_up(1)
                    by_f.setdefault(state, []).extend([sf.rate(max_new=TIER_MAX_NEW), sf.rate(max_new=TIER_MAX_NEW)])
                    say(f"    {label} {state}: {statistics.median(by_f[state][-2:]):.1f} tok/s")
                finally:
                    close(e)
            rep_f[key] = by_f
            if pick(by_f, "off") == "on":
                env[key] = on
                say(f"    {label}: more than {MIN_GAIN:.0%} faster on this PC - turned on")
                base_rate = statistics.median(by_f["on"])
        report["features"] = rep_f
    # 6. HIP: the shared-expert stream fork (a restart each), with everything chosen so far.  Both values are set
    # explicitly, so a value the config carried before does not decide the baseline.
    if hip:
        tuned = apply(base_args, settings)
        by_fork = {}
        for v in ("0", "1"):
            say(f"  Measuring the shared expert {'on its own stream' if v == '1' else 'on the main stream (the AMD default)'}"
                " (restarts the engine) ...")
            e = restart(lambda a: start_engine(a, {SH_STREAM: v}), tuned, f"{SH_STREAM}={v}", say, report)
            if e is None:
                continue
            try:
                sf = Session(e, ids_list)
                sf.warm_up(1)
                by_fork[v] = [sf.rate(), sf.rate()]
                say(f"    {statistics.median(by_fork[v]):.1f} tok/s")
            finally:
                close(e)
        report["sh_stream"] = by_fork
        if pick(by_fork, "0") == "1":
            env[SH_STREAM] = "1"
            base_rate = statistics.median(by_fork["1"])
        elif by_fork.get("0"):
            base_rate = statistics.median(by_fork["0"])
    report["seconds"] = round(time.time() - t0)
    report["tok_s"] = round(base_rate, 1) if base_rate else None
    return {"settings": settings, "env": env, "report": report}


def restart(start_engine, args, what, say, report):
    """#1337: a later step's engine start that fails is a candidate that loses, not a failed run: what the earlier
    steps measured (and the settings they chose) must survive it.  Returns the engine, or None after saying why and
    noting it in report["failed_starts"].  (The first start, for steps 1-3, is not covered: with no engine there is
    nothing measured.)"""
    try:
        return start_engine(args)
    except Exception as e:  # noqa: BLE001 - whatever the start raised (RuntimeError, EngineDied, OSError ...)
        why = str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__
        say(f"    the engine did not start for {what}: {why} - this candidate is dropped, the earlier results are kept")
        report.setdefault("failed_starts", {})[what] = why
        return None


def close(eng):
    proc = getattr(eng, "proc", None)
    if proc is None:
        return
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(60)
    except Exception:
        proc.kill()


DEFAULTS = {"--pcie-frac": None, "--spec-min-p": "0.5", "--pool-workers": None,   # None: the engine's own choice
            "--adapt-every": None, "--adapt-swaps": None, "--adapt-decay": None, "--prefill-pipe": None}


def apply(args: list[str], settings: dict) -> list[str]:
    """`args` with the calibrated settings; a setting the calibration did not change goes back to the product
    default (setup's --spec-min-p 0.5, the engine's own PCIe share and worker count), so an older calibration's
    values never linger."""
    out = list(args)
    for flag, default in DEFAULTS.items():
        out = with_arg(out, flag, settings.get(flag, default))
    return out


# our features: (engine variable, its off value, what it is called).  The variable's ON state is the engine's (or the config's) own.
FEATURES = (("STRATA_LA_LEARN", "0", "the learned look-ahead"),
            ("STRATA_ROUTE_RESIDENT", "0", "stand-ins"),
            ("STRATA_ROUTE_TAIL_SKIP", "0", "the route tail skip"),
            ("STRATA_MASK", "0", "the mask"))
# switches that are OFF by default and bit-exact (they change the speed only): measured the other way round, turned ON only when on is
# more than MIN_GAIN faster.  STRATA_POOL_DEP: the CPU pool runs a layer's gate/up and down rows as one batch with per-expert dependencies
# (no barrier between the phases); it tied on a 12 GB RTX 3060 with a 12-core CPU and may pay where the CPU pool is the bottleneck.
OPT_IN_SWITCHES = (("STRATA_POOL_DEP", "1", "the CPU pool's one-batch layer call"),)
# a feature whose variable the config does not carry is the engine's own default (tail skip, the learner's auto rule), or is not
# in use on this model (stand-ins and the mask are written by setup for the models they were built for): only the first two are
# measured without being in the config's "env"
FEATURES_WITHOUT_ENV = ("STRATA_LA_LEARN", "STRATA_ROUTE_TAIL_SKIP")
SEED_PROMPTS = 6                   # answers of TIER_MAX_NEW tokens the learners train on in this run (two rounds over PROMPTS' texts)
ENV_KEYS = (SH_STREAM,) + tuple(f[0] for f in FEATURES) + tuple(f[0] for f in OPT_IN_SWITCHES)


def apply_env(env: dict | None, calibrated: dict | None) -> dict:
    """The config's "env" with a calibration's engine variables: the ones it chose are set, the others in ENV_KEYS are
    removed (back to the engine's default), so an older calibration's choice never lingers."""
    out = {k: v for k, v in (env or {}).items() if k not in ENV_KEYS}
    out.update(calibrated or {})
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate.py <strata-*.json>")
    res = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    print(json.dumps(res, indent=1))
