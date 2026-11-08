# SPEC: the fragile re-check at near-zero decode cost (keep every re-check; make each one cheap)

*8 Nov 2026. Branch `recheck-cheap` (from combined-nov8, a55452c61). The user's rule: no feature is removed; its cost is removed.*

## What it costs today (code + 8095, calibration prompts at temperature 0)
- A cut (route_fragile_cut) keeps rows 0..r-1 of the window, discards rows r..a0 (even accepted drafts), and the next window is the
  **exact successor**: route_budget_window (verify.cpp ~4842) sets `b = 0, tail = 1e-30` and `win_exact` for the **whole window** - every
  row, every rank exact and unmasked. Such a window runs like the fragile control window: **3.6 tok/s against 38** (8 Nov), because every
  missing expert of every row is read from the SSD.
- Cuts now come every 15-25 tokens (**4.7 per 100 tokens**, k 0.012, budget 0.751; this morning 0.93 per 100 at k 0.007): calibration
  answers had 14, 23 and 20 cuts (296, 344, 583 tokens), and the slowest answer had the most.

## Objective
Same re-checks (same rows chosen, same exact decision of the re-checked row), at a decode cost <= 3% of today's all-on speed: measured
on the calibration prompts (temperature 0) as tok/s with re-check on vs the `STRATA_FRAGILE_RECHECK=0` arm, and the answers read by hand.

## Change 1 - only the re-checked row is exact
The exact successor makes **row 0 only** exact (all ranks, unmasked); rows >= 1 of that window take the normal budget and stand-ins.
- The control flag from control-windows (route_fragile_control, a host-mapped int read by route_resident_k and route_mask_k) becomes a
  **row count**: rows `< n` keep the router's picks (no stand-in move, no mask move); `n = 0` none (today's normal), `n >= T` all rows
  (today's control window). The fragile control window sets `n = T`; the exact successor sets `n = 1` and keeps the request's budget for
  the other rows (instead of `b = 0, tail = 1e-30` and the whole-window `win_exact`).
- The exact decision of row 0 is unchanged in meaning: every rank of that row exact and unmasked, as now.
- Setting `STRATA_RECHECK_ROW_ONLY=0` restores today's whole-window exact successor.

## Change 2 - the re-checked row's experts are fetched at the cut, before the successor runs
At the cut (host, between windows - no CUDA call inside a window, issue #31), the cut row's moved picks are known: the mask-learn event
list of row r (MaskEvs, kind 0 = stand-in swap, kind 1 = mask move: layer + the replaced expert), read already by mask_learn_cut.
- Queue each (layer, replaced expert) into the existing read-ahead toward VRAM (the look-ahead's prefetch path, the one
  "look-ahead tail read: ... queued" uses), highest layer-order first, so the successor's row 0 finds them on the card or in RAM.
- Bounded: at most `STRATA_RECHECK_PREFETCH_MAX` (default 96) per cut; already resident experts are skipped; a full queue drops the rest.
- Log per request: `re-check prefetch: N queued, M already resident, K were hits in the successor`.
- `STRATA_RECHECK_PREFETCH=0` turns it off.

## Change 3 - the rows after the re-checked row are kept when the exact decision agrees (stage 2, after 1 and 2 are measured)
Today rows r+1..a0 are thrown away at every cut, although most cuts do not flip (calibration: 15 of 57 cuts flipped).
Design to implement as its own step: commit rows 0..a0 as if no cut, and re-decide row r with a **one-row exact probe** that does not
step any state: the probe runs position r with the K/V cells <= r (attention is causal, so cells > r are not read) and a snapshot of
the GDN / PLE state taken just before row r inside the window (the pipeline snapshot machinery, PlSnap, holds one state per window;
this needs the state at a chosen row, saved by the window kernels when a row is marked fragile-likely by its gap before the commit).
On agreement the kept rows stand (no wasted work); on a flip the engine rewinds to row r through the existing STRATA_ROLLBACK path.
Before building: measure how often a0 > r (rows actually thrown away) and the snapshot's size, and write that step's own spec.

## Change 4 - the budget tuner sees the cuts
The tuner already charges a cut's successor time to the cut window's arm (verify.cpp ~4816, ~4986) and counts flips per arm. Add:
- per arm: cuts and cuts per 100 committed tokens, printed in the test line (`cuts 23 vs 9 (4.1 vs 1.6 /100 tok)`);
- a cut guard in the paired decision: a candidate whose cuts per token have a Wilson 90% lower bound above 2x the base's upper bound
  does not move (keep, reason "cuts").

## Tests (no server for the unit and parity parts)
1. Kernel parity (GPU, small synthetic inputs): with the row count n, rows < n return the router's picks (no move, row_shift 0) and
   rows >= n are identical to today's output; n = 0 identical to today; n = T identical to the control window.
2. Unit: the successor sets n = 1 and keeps the budget; the control window sets n = T; STRATA_RECHECK_ROW_ONLY=0 restores the old path;
   the prefetch list from a MaskEvs row (dedup, cap, resident skip); the tuner's cut guard (Wilson) and its log fields.
3. Build; serve tests; import check.
4. Live, on 8095 only when idle (the user's go is given for this work): the calibration prompts at temperature 0, arms all-on (new),
   all-on with STRATA_RECHECK_ROW_ONLY=0 STRATA_RECHECK_PREFETCH=0 (today), and STRATA_FRAGILE_RECHECK=0 (the cost floor); tok/s, cuts,
   flips, and the three answers read by hand.
