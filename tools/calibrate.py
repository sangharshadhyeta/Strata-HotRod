"""Tune the engine for this PC (setup's --calibrate).

The engine tunes itself.  While it decodes it compares settings on interleaved blocks of its own windows (include/strata/core/online_tuner.hpp):
the adaptive expert tier's swap count and decay, the draft floor (--spec-min-p) and the route tail skip.  It keeps a setting only when it is more
than 3% faster by a paired test, checks that the sum of what it adopted is no slower than the defaults, and saves what it learned per model
(<pack>/online-tune.txt); the next start begins from it.  Without a calibration this happens live, in the first minutes of use.

A calibration does the SAME thing on a schedule.  It starts the engine once with STRATA_ONLINE_TUNE=burst (smaller blocks, no waiting between
tests) and STRATA_LA_LEARN=1 (the learners train on this PC; no trained state is downloaded), sends a few real answers at a time - short
prompts and one with a context behind it - until the engine's state file says it has settled, and quits, which writes the tuner's state and the
learners' files.  No restarts: about 10-20 minutes.  (The first version tuned one setting at a time and restarted the engine about 30 times;
its picks each won alone on short answers and together decoded about 25% slower, the same setting read 19.1 or 26.3 tok/s depending on whether
its arm ran first or last, and it took 80-90 minutes - see docs/DETAILS.md.)

What stays a measurement of its own here:
  --prefill-pipe  a layer split only: the prompt chunks that keep the stages reading at once.  The best chunk count depends on b/a, the fixed
                  cost of a chunk over that of a token, which differs by an order of magnitude between rigs (about 700 tokens on four GP100
                  dies, 5000 on two RTX 3090 Ti) and costs when it is wrong (a fixed 1024 read 22-36% slower on two RTX 4090) - so it is
                  measured here, once, as the prompt read speed of fresh prompts, rather than guessed or learned while the PC serves.
  STRATA_SH_STREAM (AMD only)
                  the shared expert on its own stream: an engine variable read at start, so the tuner cannot change it live.  Off by default on
                  HIP (#816: an RX 6800 on Windows and 4x R9700 decode 12-45% faster without it), but the same card family on another PC goes
                  the other way (#1272: 2x RX 6900 XT on Linux, 6-11% faster with it), so it is measured here, per PC, with a restart per value,
                  and written into the config's "env" only when it wins.  The output is identical either way (same token ids).

    python tools/calibrate.py strata-q2_0.json        # run and print; setup.py --calibrate also saves it
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
SWEEP_ROUNDS = 3                   # the prompt chunks' paired rounds (measure_pipe)
# the adaptive tier's candidates (every, swaps, decay) against the engine's own (None): swapping more and remembering
# longer, the rest of the set as the engine has it
# HIP: the shared-expert stream fork, off (the HIP default) against on; an engine environment variable, not an argument
SH_STREAM = "STRATA_SH_STREAM"
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
    features = () if hip else ("STRATA_LA_LEARN",)     # the learners train in the same run (CUDA; the HIP engines have none)
    return measure(engine_args(cfg), ids_list, start_engine, say, doc_prompt(tok), hip=hip, features=features)


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




# ---- the calibration is the engine's online tuner, driven in a burst (HotRod, 21 Nov) ----------------------------------------------------------
# The engine settles on a good configuration by itself while it decodes (include/strata/core/online_tuner.hpp): it compares the adaptive tier's swap
# count and decay, the draft floor and the route tail skip on interleaved blocks of real windows, keeps what is more than 3% faster, and checks that
# the sum of what it adopted is no slower than the defaults.  Without a calibration that happens live, in the first minutes of use.  A calibration
# does the SAME thing on a schedule: one engine start with STRATA_ONLINE_TUNE=burst (smaller blocks, no waiting between tests) and the learners
# training (STRATA_LA_LEARN=1), a few real answers at a time until the engine's state file says "settled 1", then QUIT, which writes what the engine
# learned: the tuner's <pack>/online-tune.txt (the next start begins from it) and the learners' files.  No restarts, so about 10-20 minutes instead
# of 80-90 (the one-factor-at-a-time version restarted the engine about 30 times, and its picks added up to a loss: see docs/DETAILS.md).
ONLINE_STATE = "online-tune.txt"
ONLINE_MAX_SECONDS = 25 * 60       # the burst's time cap; it stops earlier when the engine reports settled
ONLINE_MAX_ROUNDS = 120            # ... and after this many rounds (each round answers every prompt once)
ONLINE_LONG_TOKENS = 3000          # the context of the burst's long prompt (real use has long contexts; the tuner must see their windows too)
ENV_KEYS = (SH_STREAM, "STRATA_LA_LEARN", "STRATA_ROUTE_RESIDENT", "STRATA_ROUTE_TAIL_SKIP", "STRATA_MASK", "STRATA_POOL_DEP")   # older choices reset


def online_state_path(args: list[str]) -> Path | None:
    pack = arg_value(args, "--pack")
    return Path(pack) / ONLINE_STATE if pack else None


def read_online_state(path: Path | None) -> dict:
    """The engine's saved tuner state as {knob: level, "settled": 0|1}; empty when there is none."""
    out: dict = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines() if path else []:
            p = line.split()
            if len(p) == 3 and p[0] == "knob":
                out[p[1]] = float(p[2])
            elif len(p) == 2 and p[0] == "settled":
                out["settled"] = int(p[1])
    except (OSError, ValueError):
        return {}
    return out


def measure(base_args: list[str], ids_list, start_engine, say=print, long_prompt=None, hip: bool = False, features=(),
            max_seconds: float = ONLINE_MAX_SECONDS, max_rounds: int = ONLINE_MAX_ROUNDS) -> dict:
    """One engine start in the tuner's burst mode.  `features`: the engine switches to learn (STRATA_LA_LEARN: the look-ahead's learner)."""
    t0 = time.time()
    report: dict = {}
    base_args = apply(base_args, {})                   # the product defaults: what the tuner starts from and must beat
    state = online_state_path(base_args)
    if state is not None:
        try:
            state.unlink()                             # a calibration learns this PC afresh
        except OSError:
            pass
    env = {"STRATA_ONLINE_TUNE": "burst"}
    if "STRATA_LA_LEARN" in features:
        env["STRATA_LA_LEARN"] = "1"
    say("  Loading the model; the engine tunes itself on real answers (a few minutes) ...")
    eng = start_engine(base_args, env)
    pipe = 0
    rounds = 0
    try:
        info = dict(getattr(eng, "info", {}) or {})
        s = Session(eng, ids_list)
        s.warm_up()
        # 0. a layer split's prompt chunks (prompt read speed; independent of the decode settings the tuner owns)
        stages = info.get("prefill_stages")
        split = stages > 1 if isinstance(stages, int) else "--layer-split" in base_args
        if long_prompt is not None and split:
            say("  Measuring the prompt chunks of the layer split ...")
            pipe, rep_pipe = measure_pipe(eng, long_prompt, say)
            report.update(rep_pipe)
        prompts = list(ids_list)
        if long_prompt is not None:
            try:
                prompts.append(long_prompt(0, ONLINE_LONG_TOKENS))
            except Exception:  # noqa: BLE001 - a prompt source that cannot make one: the short prompts only
                pass
        sb = Session(eng, prompts)
        rates = []
        while rounds < max_rounds and time.time() - t0 < max_seconds:
            rates.append(sb.rate(max_new=TIER_MAX_NEW))
            rounds += 1
            st = read_online_state(state)
            if rounds % 4 == 0:
                say(f"    round {rounds}: {rates[-1]:.1f} tok/s" + ("; the engine reports its settings settled" if st.get("settled") else ""))
            if st.get("settled"):
                break
        report["rates"] = [round(r, 1) for r in rates]
    finally:
        close(eng)                                     # QUIT: the engine writes the tuner's state and the learners' files
    learned = read_online_state(state)
    report["tuner"] = learned
    report["settled"] = bool(learned.get("settled"))
    report["rounds"] = rounds
    if not learned:
        say("    the engine wrote no tuner state (its tuner is off here, or this engine has none): nothing to keep")
    elif not report["settled"]:
        say("    the time cap came before the tuner settled: what it has is kept, and it carries on live")
    else:
        say("    settled: " + ", ".join(f"{k} {v:g}" for k, v in learned.items() if k != "settled"))
    settings = {}
    if pipe:
        settings["--prefill-pipe"] = str(int(pipe))
    env_out: dict = {}
    base_rate = statistics.median(report["rates"][-3:]) if report.get("rates") else None
    # HIP: the shared-expert stream fork (a restart each), the one setting the tuner cannot change while running.  Both values are set
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
            env_out[SH_STREAM] = "1"
            base_rate = statistics.median(by_fork["1"])
        elif by_fork.get("0"):
            base_rate = statistics.median(by_fork["0"])
    report["seconds"] = round(time.time() - t0)
    report["tok_s"] = round(base_rate, 1) if base_rate else None
    return {"settings": settings, "env": env_out, "report": report}



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
        proc.wait(180)     # a big model saves its learned files and frees tens of GB of pinned memory on the way out
    except Exception:
        proc.kill()
        try:
            proc.wait(120)   # 21 Nov: a kill that is not waited for let the next restart start while this engine's pinned memory was still
        except Exception:    # being released: two engines at once, and the memory scope killed one (OOM at 24.9 GB resident in a 29 GB scope)
            pass


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
