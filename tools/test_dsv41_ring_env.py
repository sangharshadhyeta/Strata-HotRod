"""The DeepSeek engine's ring, lend and stand-in switches that setup writes into a fresh install's config (setup.dsv41_ring_env), and
the re-run rule that keeps the person's own env entries.

    python -m unittest tools.test_dsv41_ring_env
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

ONE = [{"vendor": "nvidia", "uma": False, "index": 0}]
SIX = {"DS4_RING", "DS4_RING_PIN", "STRATA_STANDIN", "DS4_LEND", "DS4_LEND_RING", "DS4_RING_STICKY"}


class RingEnv(unittest.TestCase):
    def env(self, cards, **ev):
        with mock.patch.dict(os.environ, ev, clear=False):
            for k in ("STRATA_DSV41_RING",):
                if k not in ev:
                    os.environ.pop(k, None)
            return setup.dsv41_config_env(cards)

    def test_one_nvidia_card_gets_the_six_switches(self):
        e = self.env(ONE)
        self.assertEqual(set(e), SIX)
        self.assertEqual(e["DS4_RING"], "32")

    def test_other_setups_get_none(self):
        self.assertEqual(self.env(ONE * 2), {})
        self.assertEqual(self.env([{"vendor": "amd", "uma": False}]), {})
        self.assertEqual(self.env([]), {})

    def test_a_unified_memory_apu_keeps_its_own_setting(self):
        self.assertEqual(self.env([{"vendor": "amd", "uma": True}]), {"DS4_ASYNC_T": "4"})

    def test_the_flag_turns_it_off_or_sets_the_ceiling(self):
        self.assertEqual(self.env(ONE, STRATA_DSV41_RING="0"), {})
        self.assertEqual(self.env(ONE, STRATA_DSV41_RING="x"), {})
        self.assertEqual(self.env(ONE, STRATA_DSV41_RING="16")["DS4_RING"], "16")

    def test_a_rerun_refreshes_setups_keys_and_keeps_the_persons_own(self):
        old = {"arch": "deepseek41", "env": {"DS4_RING": "48", "STRATA_STANDIN": "1", "MY_VAR": "1"}}
        new = {"arch": "deepseek41", "env": {}, "model_name": "m"}
        setup.carry_over(old, new)
        self.assertEqual(new["env"], {"MY_VAR": "1"})   # (the ring is off on this run: its old keys do not linger)

    def test_another_models_standin_key_is_the_persons_own(self):
        old = {"arch": "qwen35moe", "env": {"STRATA_STANDIN": "1"}}
        new = {"arch": "qwen35moe", "env": {}, "model_name": "m"}
        setup.carry_over(old, new)
        self.assertEqual(new["env"], {"STRATA_STANDIN": "1"})


if __name__ == "__main__":
    unittest.main()
