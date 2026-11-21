"""Tests for tools/calibrate.py and setup's use of it, without a GPU: a stand-in engine whose decode speed is a
function of the settings it runs with.

    python -m unittest tools.test_calibrate
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate as CAL  # noqa: E402


class FakeEngine:
    """Speed = f(pcie_frac, spec_min_p, workers): the GEN line's tune keys arrive as `strata_tune`."""

    def __init__(self, args, speed, info_workers=6, starts=None, adapt=None, read=None, reads=None):
        self.args = list(args)
        self.read, self.reads = read, reads      # prompt read speed = read(prefill_pipe); the values it read with
        self.adapt_bonus = (adapt or {}).get(CAL.arg_value(args, "--adapt-swaps"), 1.0)
        w = CAL.arg_value(args, "--pool-workers")
        self.workers = int(w) if w else info_workers
        self.info = {"pool_workers": info_workers, "pcie_frac": 0.55, "spec_min_p": float(CAL.arg_value(args, "--spec-min-p") or 0)}
        self.speed = speed
        self.last = {}
        self.proc = None
        if starts is not None:
            starts.append(list(args))

    def generate(self, ids, max_new, sampling, cancel):
        tune = sampling.get("strata_tune") or {}
        self.max_new = max_new
        arg_f = CAL.arg_value(self.args, "--pcie-frac")
        rate = self.speed(tune.get("pcie_frac", float(arg_f) if arg_f else 0.55), tune.get("spec_min_p", self.info["spec_min_p"]), self.workers)
        rate *= self.adapt_bonus
        for _ in range(max_new):
            yield 1
        self.last = {"generated": max_new, "decode_ms": max_new / rate * 1000.0}
        if "prefill_pipe" in tune and self.read is not None:
            n = len(ids)
            self.last.update(prompt_tokens=n, prompt_read=n, prompt_ms=n / self.read(tune["prefill_pipe"]) * 1000.0)
            if self.reads is not None:
                self.reads.append(tune["prefill_pipe"])


BASE = ["--pack", "p", "--spec", "4", "--spec-min-p", "0.5", "--max-context", "8192"]


class Calibrate(unittest.TestCase):
    def run_with(self, speed, workers=6, base=BASE, adapt=None, read=None, reads=None):
        starts = []
        res = CAL.measure(base, [[1, 2, 3]] * 3, lambda a, env=None: FakeEngine(a, speed, workers, starts, adapt, read, reads),
                          say=lambda *_: None, long_prompt=lambda i, n: [1] * n, max_rounds=2)
        return res, starts

    def test_prefill_pipe_on_a_layer_split(self):
        # two stages that read fastest with b/a 1536: kept, and only one engine start measures it
        split = BASE + ["--layer-split", "auto"]
        reads = []
        res, _ = self.run_with(lambda f, p, w: 50.0, base=split, reads=reads,
                               read=lambda v: {1536: 600.0, 768: 560.0, 1: 520.0}.get(v, 500.0))
        self.assertEqual(res["settings"].get("--prefill-pipe"), "1536")
        self.assertEqual(sorted(set(reads)), sorted(set(CAL.PIPE_VALUES)))
        self.assertIn("pipe_sweep", res["report"])
        self.assertEqual(CAL.arg_value(CAL.apply(split, res["settings"]), "--prefill-pipe"), "1536")

    def test_prefill_pipe_noise_stays_off(self):
        # 2% faster evened out: below MIN_GAIN, so the engine default (off) stays
        res, _ = self.run_with(lambda f, p, w: 50.0, base=BASE + ["--layer-split", "auto"],
                               read=lambda v: 510.0 if v == 1 else 500.0)
        self.assertNotIn("--prefill-pipe", res["settings"])

    def test_prefill_pipe_must_win_every_round(self):
        # 1536 looks 12% faster in the sweep and the median, but one paired round reads slower: noise, it stays off
        calls = {}

        def read(v):
            calls[v] = calls.get(v, 0) + 1
            if v != 1536:
                return 500.0
            return [560.0, 560.0, 560.0, 480.0, 560.0, 560.0][min(calls[v], 6) - 1]
        res, _ = self.run_with(lambda f, p, w: 50.0, base=BASE + ["--layer-split", "auto"], read=read)
        self.assertNotIn("--prefill-pipe", res["settings"])
        self.assertEqual(len(res["report"]["pipe_confirm"]["1536"]), 3)

    def test_prefill_pipe_reads_the_same_text(self):
        # every read: the same stretch of text after a header of its own (only the chunks differ between values)
        tok = type("T", (), {"encode": lambda s, t, **k: [ord(c) for c in t]})()
        make = CAL.doc_prompt(tok)
        a, b = make(1, 300), make(2, 300)
        self.assertNotEqual(a[:40], b[:40])
        self.assertEqual(a[-320:], b[-320:])

    def test_prefill_pipe_only_on_a_layer_split(self):
        reads = []
        res, _ = self.run_with(lambda f, p, w: 50.0, read=lambda v: 600.0 if v == 1536 else 500.0, reads=reads)
        self.assertEqual(reads, [])
        self.assertNotIn("--prefill-pipe", res["settings"])
        # an old calibration's value goes when a new one does not keep it
        self.assertIsNone(CAL.arg_value(CAL.apply(BASE + ["--prefill-pipe", "800"], {}), "--prefill-pipe"))

    def test_prefill_pipe_follows_the_engines_stage_count(self):
        # --layer-split that left one prompt stage (the engine says prefill_stages=1): nothing to measure
        reads = []

        class OneStage(FakeEngine):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.info["prefill_stages"] = 1
        res = CAL.measure(BASE + ["--layer-split", "auto"], [[1, 2, 3]] * 3,
                          lambda a, env=None: OneStage(a, lambda f, p, w: 50.0, 6, None, None,
                                                       lambda v: 600.0 if v == 1536 else 500.0, reads),
                          say=lambda *_: None, long_prompt=lambda i, n: [1] * n, max_rounds=2)
        self.assertEqual(reads, [])
        self.assertNotIn("--prefill-pipe", res["settings"])

    def test_sh_stream_is_measured_on_hip_only(self):
        # #1272 / #816: the shared-expert fork is measured off (the HIP default) against on, each with a restart and
        # the engine variable set explicitly; it is kept only when it wins by more than MIN_GAIN
        def run(bonus, hip=True):
            starts, envs = [], []

            def start(args, env=None):
                envs.append(dict(env or {}))
                fork = (env or {}).get(CAL.SH_STREAM) == "1"
                return FakeEngine(args, lambda f, p, w: 50.0 * (bonus if fork else 1.0), 6, starts)
            res = CAL.measure(BASE, [[1, 2, 3]] * 3, start, say=lambda *_: None, hip=hip, max_rounds=2)
            return res, starts, envs
        res, starts, envs = run(1.08)
        self.assertEqual(res["env"], {CAL.SH_STREAM: "1"})
        fork_envs = [e for e in envs if CAL.SH_STREAM in e]
        self.assertEqual(fork_envs[:2], [{CAL.SH_STREAM: "0"}, {CAL.SH_STREAM: "1"}])
        self.assertEqual(starts[-1], starts[-2])                      # the same args for both: only the variable differs
        self.assertGreater(res["report"]["tok_s"], 52.0)              # the reported speed is the fork's
        flat, _, _ = run(1.02)                                        # within noise: the default stays
        self.assertEqual(flat["env"], {})
        self.assertEqual(set(flat["report"]["sh_stream"]), {"0", "1"})
        slower, _, _ = run(0.80)
        self.assertEqual(slower["env"], {})
        cuda, _, envs = run(1.50, hip=False)                          # not AMD: no step, no engine variable
        self.assertEqual(cuda["env"], {})
        self.assertNotIn("sh_stream", cuda["report"])
        self.assertTrue(all(CAL.SH_STREAM not in e for e in envs))

    def test_feature_keys_do_not_linger(self):
        old = {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ROUTE_RESIDENT": "0", "STRATA_MASK": "learn", "OTHER": "1"}
        out = CAL.apply_env(old, {"STRATA_ROUTE_TAIL_SKIP": "0"})
        self.assertEqual(out, {"OTHER": "1", "STRATA_ROUTE_TAIL_SKIP": "0"})   # an older calibration's choices are reset first

    def test_the_burst_runs_the_engines_tuner_until_it_settles(self):
        # one engine start with the tuner's burst schedule and the learners training; real answers until the engine's state file says
        # settled; no restarts; what the engine learned is reported, and the settings are not rewritten from outside
        with tempfile.TemporaryDirectory() as pack:
            state = Path(pack) / CAL.ONLINE_STATE
            state.write_text("knob adapt_swaps 99\nsettled 1\n")      # an older state: a calibration learns afresh, so it is removed first
            envs, calls = [], {"n": 0}

            class Tuning(FakeEngine):
                def generate(self, ids, max_new, sampling, cancel):
                    calls["n"] += 1
                    if calls["n"] == 3 * 4:                           # after four rounds of three prompts the engine has settled
                        state.write_text("knob adapt_swaps 96\nknob spec_min_p 0.5\nsettled 1\n")
                    return super().generate(ids, max_new, sampling, cancel)

            def start(args, env=None):
                envs.append(dict(env or {}))
                self.assertFalse(state.exists() and calls["n"] == 0, "the old state must be gone before the engine starts")
                return Tuning(args, lambda f, p, w: 50.0, 6)
            res = CAL.measure(BASE[2:] + ["--pack", pack], [[1, 2, 3]] * 3, start, say=lambda *_: None, features=("STRATA_LA_LEARN",))
            self.assertEqual(len(envs), 1)                             # one start, no restarts
            self.assertEqual(envs[0], {"STRATA_ONLINE_TUNE": "burst", "STRATA_LA_LEARN": "1"})
            self.assertTrue(res["report"]["settled"])
            self.assertEqual(res["report"]["tuner"], {"adapt_swaps": 96.0, "spec_min_p": 0.5, "settled": 1})
            self.assertLessEqual(res["report"]["rounds"], 6)
            self.assertEqual(res["settings"], {})
            self.assertEqual(res["env"], {})

    def test_the_burst_stops_at_its_cap_when_nothing_settles(self):
        with tempfile.TemporaryDirectory() as pack:
            res = CAL.measure(["--pack", pack], [[1, 2, 3]] * 3, lambda a, env=None: FakeEngine(a, lambda f, p, w: 50.0, 6),
                              say=lambda *_: None, max_rounds=3)
            self.assertFalse(res["report"]["settled"])
            self.assertEqual(res["report"]["rounds"], 3)
            self.assertEqual(res["report"]["tuner"], {})

    def test_online_state_is_read_tolerantly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text("knob a 1.5\nbroken line\nknob b notanumber\nsettled 1\n")
            self.assertEqual(CAL.read_online_state(p), {})            # a malformed file is no state
            p.write_text("knob a 1.5\nsettled 0\n")
            self.assertEqual(CAL.read_online_state(p), {"a": 1.5, "settled": 0})
        self.assertEqual(CAL.read_online_state(None), {})

    def test_apply_env(self):
        old = {"STRATA_ROCBLAS_TUNING": "tools/hip", CAL.SH_STREAM: "1"}
        self.assertEqual(CAL.apply_env(old, {}), {"STRATA_ROCBLAS_TUNING": "tools/hip"})   # an older choice goes
        self.assertEqual(CAL.apply_env({"X": "1"}, {CAL.SH_STREAM: "1"}), {"X": "1", CAL.SH_STREAM: "1"})
        self.assertEqual(CAL.apply_env(None, None), {})

    def test_apply(self):
        a = CAL.apply(BASE, {"--pcie-frac": "0.35", "--pool-workers": "4"})
        self.assertEqual(CAL.arg_value(a, "--pcie-frac"), "0.35")
        self.assertEqual(CAL.arg_value(a, "--pool-workers"), "4")
        self.assertEqual(CAL.arg_value(a, "--spec-min-p"), "0.5")
        b = CAL.apply(a, {})                                # back to the defaults
        self.assertIsNone(CAL.arg_value(b, "--pcie-frac"))
        self.assertIsNone(CAL.arg_value(b, "--pool-workers"))
        self.assertEqual(b.count("--spec-min-p"), 1)

    def test_engine_args_are_the_servers(self):
        # #447: a "gpu" list is a layer split only with two or more cards; "0,2" is one too; split_skip_if_fits counts
        helper = {"args": BASE + ["--expert-cache-device1", "1800"], "gpu": [0], "env": {"CUDA_VISIBLE_DEVICES": "0,1"}}
        self.assertEqual(CAL.engine_args(helper), helper["args"])                 # one stage + a helper card: no split
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": 1}), BASE)
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": [0, 2]}), BASE + ["--layer-split", "auto"])
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": "0,2", "layer_split": "18"}),
                         BASE + ["--layer-split", "18"])
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": [0, 1], "split_skip_if_fits": True}),
                         BASE + ["--layer-split", "auto", "--split-skip-if-fits"])
        own = BASE + ["--layer-split", ""]                                       # the config's own value wins
        self.assertEqual(CAL.engine_args({"args": own, "gpu": [0, 1]}), own)

    def test_run_measures_with_the_servers_args(self):
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d)
            (tok / "vocab.json").write_text(json.dumps({"a": 0, "b": 1}))
            (tok / "merges.txt").write_text("")
            (tok / "token_type.json").write_text(json.dumps([1, 1]))
            seen = []
            saved = CAL.measure
            CAL.measure = lambda args, ids_list, start_engine, say=print, long_prompt=None, hip=False, features=(): (
                seen.append((args, hip)) or {})
            fake = type("ST", (), {"Tokenizer": lambda *a: type("T", (), {"encode": lambda s, t, **k: [0]})()})
            try:
                with mock.patch.dict(sys.modules, {"strata_tokenizer": fake}):
                    CAL.run({"tokenizer": d, "args": list(BASE), "gpu": [0]}, start_engine=lambda a: None)
                    CAL.run({"tokenizer": d, "args": list(BASE), "gpu": [0], "backend": "hip"}, start_engine=lambda a: None)
            finally:
                CAL.measure = saved
        self.assertEqual(seen, [(BASE, False), (BASE, True)])         # the shared-expert step runs on AMD only

    def test_engine_error(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata-x.log"
            log.write_text("strata generate: an old error\n", encoding="utf-8")
            since = log.stat().st_size
            with open(log, "a", encoding="utf-8") as f:
                f.write("loading ...\nstrata generate: --expert-cache-remote with a layer split needs a GPU that runs "
                        "no stage (2 visible, 2 used by the split)\n\n")
            self.assertEqual(CAL.engine_error(str(log), since), "strata generate: --expert-cache-remote with a layer "
                             "split needs a GPU that runs no stage (2 visible, 2 used by the split)")
            with open(log, "a", encoding="utf-8") as f:
                f.write("Segmentation fault\n")
            self.assertIn("--expert-cache-remote", CAL.engine_error(str(log), since))   # the engine's own line first
            self.assertEqual(CAL.engine_error(str(log), log.stat().st_size), None)   # nothing new since
            self.assertIsNone(CAL.engine_error(None))
            self.assertIsNone(CAL.engine_error(str(Path(d) / "missing.log")))

    def test_pick(self):
        self.assertEqual(CAL.pick({"a": [50, 51, 49], "b": [53, 52, 60]}, "a"), "b")
        self.assertEqual(CAL.pick({"a": [50, 51, 49], "b": [51, 51.5, 51]}, "a"), "a")
        self.assertEqual(CAL.pick({}, "a"), "a")


class SetupIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
        os.environ["APPDATA"] = self.tmp.name
        os.environ["XDG_CONFIG_HOME"] = self.tmp.name
        import setup as S
        self.S = S
        self.saved = (S.gpu_info, S.cpu_info, S.ram_gb, CAL.run)
        S.gpu_info = lambda pick=None: {"name": "RTX Test", "vram_gb": 12.0, "arch": "120", "count": 1, "index": 0}
        S.cpu_info = lambda: ("Test CPU", True, True)
        S.ram_gb = lambda: 64.0

    def tearDown(self):
        self.S.gpu_info, self.S.cpu_info, self.S.ram_gb, CAL.run = self.saved
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_calibrate_config_writes_and_remembers(self):
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg = {"exe": "x", "args": list(BASE), "model_name": "qwen3.8-flash-next-q2_0"}
        cfg_path.write_text(json.dumps(cfg))
        CAL.run = lambda c, say=print, start_engine=None: {"settings": {"--pcie-frac": "0.20"}, "report": {"tok_s": 61.2}}
        self.assertTrue(self.S.calibrate_config(cfg_path))
        written = json.loads(cfg_path.read_text())
        self.assertEqual(CAL.arg_value(written["args"], "--pcie-frac"), "0.20")
        saved = self.S.saved_calibration(written)
        self.assertEqual(saved["settings"], {"--pcie-frac": "0.20"})
        # another model on the same PC has no calibration yet
        self.assertIsNone(self.S.saved_calibration({**written, "model_name": "swift-iq2_xs"}))
        # another context size is another key (its KV cache changes the expert cache)
        other = dict(written, args=CAL.with_arg(written["args"], "--max-context", "131072"))
        self.assertIsNone(self.S.saved_calibration(other))

    def test_calibrate_config_writes_the_env_choice(self):
        # HIP: the measured engine variable goes into the config's "env" and the settings file; a re-run that finds it
        # no longer faster removes it again, and leaves the config's other variables alone
        cfg_path = Path(self.tmp.name) / "strata-iq3_s.json"
        cfg = {"exe": "x", "args": list(BASE), "model_name": "m", "backend": "hip", "env": {"STRATA_ROCBLAS_TUNING": "t"}}
        cfg_path.write_text(json.dumps(cfg))
        CAL.run = lambda c, say=print, start_engine=None: {"settings": {}, "env": {CAL.SH_STREAM: "1"}, "report": {"tok_s": 70.1}}
        self.assertTrue(self.S.calibrate_config(cfg_path))
        written = json.loads(cfg_path.read_text())
        self.assertEqual(written["env"], {"STRATA_ROCBLAS_TUNING": "t", CAL.SH_STREAM: "1"})
        self.assertEqual(self.S.saved_calibration(written)["env"], {CAL.SH_STREAM: "1"})
        CAL.run = lambda c, say=print, start_engine=None: {"settings": {}, "env": {}, "report": {"tok_s": 65.0}}
        self.assertTrue(self.S.calibrate_config(cfg_path))
        self.assertEqual(json.loads(cfg_path.read_text())["env"], {"STRATA_ROCBLAS_TUNING": "t"})

    def test_failed_calibration_keeps_defaults(self):
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg_path.write_text(json.dumps({"exe": "x", "args": list(BASE), "model_name": "m"}))

        def boom(*a, **k):
            raise RuntimeError("the engine exited before it was ready")
        CAL.run = boom
        self.assertFalse(self.S.calibrate_config(cfg_path))
        self.assertEqual(json.loads(cfg_path.read_text())["args"], BASE)
        self.assertIsNone(self.S.saved_calibration({"args": BASE, "model_name": "m"}))

    def test_failed_calibration_says_the_engines_reason(self):
        # #447: the engine's last error line is printed, and the --calibrate run repeats the failure before it starts
        import contextlib
        import io
        log = Path(self.tmp.name) / "strata-q2_0.log"
        log.write_text("strata generate: an error of an earlier start\n", encoding="utf-8")
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg_path.write_text(json.dumps({"exe": "x", "args": list(BASE), "model_name": "m", "log": str(log)}))

        def boom(*a, **k):
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata generate: --expert-cache-remote with a layer split needs a GPU that runs no stage\n")
            raise RuntimeError(f"the engine exited before it was ready (see {log})")
        CAL.run = boom
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertFalse(self.S.calibrate_config(cfg_path))
        self.assertIn("the engine said: strata generate: --expert-cache-remote with a layer split", out.getvalue())
        self.assertNotIn("an earlier start", out.getvalue())
        out = io.StringIO()
        with mock.patch.object(self.S, "ROOT", Path(self.tmp.name)), \
                mock.patch.object(self.S, "data_folder", lambda d: (Path(self.tmp.name) / "data", [])), \
                mock.patch.object(self.S, "update_installed_engine", lambda *a: None), \
                mock.patch.object(sys, "argv", ["setup.py", "--calibrate", "--no-start"]), \
                contextlib.redirect_stdout(out):
            self.assertEqual(self.S.main(), 0)
        self.assertIn("this PC is NOT tuned: the tuning failed (the reason is above); the model keeps the default "
                      "settings", out.getvalue())


if __name__ == "__main__":
    unittest.main()
