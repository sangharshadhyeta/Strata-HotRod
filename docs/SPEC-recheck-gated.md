# SPEC: the re-check gated by drafter confidence (prevent the flips; recompute only where it pays)

*8 Nov 2026. Branch `recheck-gated` (from combined-nov8 4e1693cd1, which has recheck-cheap: the row-only exact successor, prefetch at
the cut, cuts in the tuner). The pipeline stays whole; the drafter's confidence decides, per row, what each row gets.*

## The data (8095, 18 requests of mixed code / reasoning / prose, the "re-check by draft confidence" counters)
| drafter p of the row's token | rows moved by stand-ins | cuts | flips | flips per moved row |
| --- | --- | --- | --- | --- |
| 0.5-0.9 (4 buckets) | 1,225 | 165 | 28 | 2.3% |
| 0.9-1.0 | 3,607 | 80 | 8 | 0.22% |
| none (no draft for the row's output: mostly the window's last row) | 3,963 | 524 | 114 | 2.9% |
Drafts below 0.5 are never made (--spec-min-p 0.5). Confident rows flip 10x less; "none" rows hold 76% of the flips.

## The rule (per row, decided on the host before the window runs, from dprob)
1. **Confident drafted rows (p >= P_HI, default 0.9):** stand-ins as today; **not eligible for a cut** (no re-check), except in an
   audit window (below). 
2. **Uncertain drafted rows (p < P_HI):** **exact up front** - the row keeps the router's picks (no stand-in move, no mask move), so
   it cannot flip and needs no re-check. Uses a per-row mask (step A).
3. **"none" rows (no drafter p):** as today with recheck-cheap - stand-ins, eligible for a cut, the row-only exact successor and the
   prefetch at the cut. (Step C measures whether their early router margin can move them to rule 2 later.)
4. **Audit:** every `STRATA_RECHECK_AUDIT_EVERY` decode windows (default 256) the rule is suspended for one window: every row as today
   (stand-ins, all rows eligible for a cut). Its counters per bucket keep the flip rate of confident rows measured.
5. **The threshold learns:** P_HI moves one bucket down (more rows confident) when the audit's flips per moved row of the bucket just
   below P_HI has a Wilson 90% upper bound <= 0.5%, and one bucket up when the confident buckets' audit flip rate has a Wilson lower
   bound > 0.5%; at most one move per 2,000 audited moved rows; range 0.6-0.95; kept in the stand-in tune file per model.
6. Settings: `STRATA_RECHECK_GATE=0` restores today's behaviour (recheck-cheap) exactly; `STRATA_RECHECK_P_HI` sets the start value.

## Step A - a per-row hold mask
The fragile flag is a row count today (recheck-cheap: rows < n are held). Make it a **row bitmask** (u32, rows 0..31; the window has
at most S <= 8 rows): bit r set = row r keeps the router's picks in route_resident_k and route_mask_k. The successor's row 0 = bit 0;
the control window = all bits; rule 2's rows = their bits. Host-mapped, written between windows (no host CUDA call inside a window).
Parity: mask 0 = today's normal path; mask with low n bits = today's row count n; all bits = the control window.

## Step B - eligibility
route_fragile_first takes an eligibility per row (host array): rows of rule 1 (outside an audit window) and rule 2 are not eligible;
the stop re-check (a stop id at a moved row) stays for every moved row that is eligible.

## Step C - the early router margin, counted (measurement only)
For each row, the minimum over layers 0-3 of the router's top-1 minus top-2 logit (the routed layer's own router, before stand-ins),
written per row into a small device array by the routing kernel of those layers (predicated store, graph-safe), read with the gaps
after the window (that read exists already). Counted for "none" rows: moved / cut / flipped per margin decile (deciles from a running
estimate), printed per request: `re-check by early margin (none rows, deciles 0-9): moved ...; cut ...; flipped ...`.

## Counters and logs
Per request: rows held up front (rule 2), cuts skipped by rule 1, audit windows and their per-bucket moved / cut / flipped, P_HI.

## Tests
1. Kernel parity (GPU): the bitmask cases above against today's kernels (exact equality of ids, weights, row_shift).
2. Unit (CPU): the per-row rule from dprob (buckets, P_HI, none rows), eligibility, the audit cadence, the threshold mover (scripted
   counts, Wilson bounds, range, one move per 2,000).
3. Build; serve tests; import check.
4. Live on 8095 (idle only; the user's go is given): the calibration prompts at temperature 0, gate on vs STRATA_RECHECK_GATE=0 vs
   STRATA_FRAGILE_RECHECK=0, twice each; tok/s, flips caught by the audit, held rows; answers read by hand.

## Step F - the per-row shift cap (prevent the drift that flips; added 8 Nov after counter D)
Counter D on 8095 (7 requests): the flipped share of cut rows' swaps is the same at every rank (39.1-39.9%, ranks 3-9) and every layer
band (39.3-39.7%) - a flip is the sum of many small shifts in one row, not one bad stand-in. So the row's accumulated shift is capped
while the row is computed:
- **The cap:** route_resident_k (and route_mask_k for mask moves) add each move's risk to row_shift[row]. A move is taken only if
  row_shift[row] + this move's shift <= cap[row]; otherwise the pick stays the router's own (exact, as a held row's pick). One extra
  comparison per candidate move. cap[] is a host-mapped float per row (<= 32 rows), written between windows (graph-safe, no host CUDA
  call inside a window); +inf = today's behaviour.
- **The cap per row from its class** (rules 1-3): confident rows (p >= P_HI) `cap_conf` (default +inf), uncertain rows `cap_unc`, none
  rows `cap_none`. Defaults are multiples of a running median of moved rows' final row_shift (on the host, from the route_fragile_shifts
  read that exists): cap_unc = 1.5 x median, cap_none = 1.0 x median. Settings: STRATA_SHIFT_CAP=0 (off: +inf everywhere),
  STRATA_SHIFT_CAP_UNC / STRATA_SHIFT_CAP_NONE (the multipliers), STRATA_SHIFT_CAP_CONF (multiplier, default inf).
- **Calibration counter:** for cut rows, the final row_shift by decile (running deciles), split flipped / not; one request line
  `re-check by row shift (deciles 0-9, flipped / not): ...` - shows where the flips sit against the caps.
- **Per request:** `shift cap: moves refused N (rows capped M), caps conf/unc/none x/y/z`.
- The re-check, the audit and the gate stay (the safety net).
Tests: kernel parity (cap +inf identical to today; a finite cap refuses exactly the moves past it; row_shift never exceeds the cap; held
rows unchanged), CPU unit tests (the median, the multipliers, the classes), build, serve tests, import check.
