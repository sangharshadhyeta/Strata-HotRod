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
- **Opt-in since 9 Nov (`STRATA_SHIFT_CAP=1`).** Two frozen-state A/Bs on 8095 had the cap off faster in every run (+3.5%, +3.7%):
  each refused move is an exact pick and a read, and the re-check on flip already guards the rows the cap protects.
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

## Step G - the shift-cap multipliers and the prefetch count chosen live (8 Nov, the user)
The paired tuner (include/strata/core/paired_test.hpp; the budget test and the mask test in verify.cpp) gets two more tests in its
rotation: budget -> mask -> **cap** -> **prefetch** -> budget ...
- **Cap test:** the candidate scales both multipliers (cap_unc, cap_none) by the test's step (x or / 1.25 by direction, the same
  direction / step rules as the budget test); arms interleaved and paired as today; decided by paired::decide with the quality veto and
  the cut guard. A move keeps the ratio cap_unc / cap_none. Range 0.25-8 x the median; +inf (off) is not a candidate.
- **Prefetch test:** the candidate changes STRATA_RECHECK_PREFETCH_MAX by x or / 1.5 (rounded, at least 1 step), range 0-64; 0 is a
  candidate (no prefetch). Same decision rule.
- The arm's values are written between windows (the cap array, the prefetch count on the host); nothing inside a window.
- The chosen values persist in the .gate state file with the rest (loaded at start; the env values are the start values only when no
  saved state exists).
- Logs: `auto cap test x vs y: pairs N, mean d ..., -> keep|move (reason)` and the same for `auto prefetch test`.
- Settings: STRATA_TUNE_CAP=0, STRATA_TUNE_PREFETCH=0 leave them fixed.
Tests: CPU unit tests of the rotation, the candidates (steps, ranges, ratio kept), the persistence round trip; build; serve tests.

## Step H - k from evidence above the noise floor, and its margin chosen by the tuner (8 Nov evening, the user: "fix k with the tuner")
**What is wrong (verify.cpp route_fragile_settle ~5420-5448; 8095 log):** every 50 re-checks `k = 1.25 x largest flipped ratio` in the
ring of the last 200 re-checks. About 7% of exact-vs-exact control re-checks flip (control upper bound 0.072-0.074 over 34-35
samples), at any ratio, so noise flips at high ratios set `largest`. The only noise guard compares the ring's OVERALL flip rate with
the control's (ring_lo 0.12-0.16 > ctrl_hi 0.07 passes because low-ratio flips dominate), so k kept rising: 0.012 -> 0.067. Result: 63-83
cuts per answer, the re-check costing ~30% of decode.

**H1 - k_safe, noise-aware (the quality base).** Replace `largest` by `r*`: with the ring's re-checks sorted by ratio, the LARGEST flipped
ratio t such that, among ring entries with ratio >= t, n_t >= 8 and the flips f_t have Wilson-90% lower bound > the control upper bound
(`route_ctrl_bounds`; with fewer than 20 control samples the bound is 1, so no raise, as today). `k_safe = clamp(1.25 x r*, 0.005, 50)`.
If no t qualifies and the ring has >= 50 entries: target = 0.5 x old (the existing step limits stay: up to 1.5x, down to 0.5x per update).
Pure function in recheck_gate.hpp (`k_evidence(ratios, flips, n, ctrl_hi)`), unit-tested: a ring whose only high-ratio flips are at the
noise rate gives a low k_safe; a ring whose flips concentrate below ratio 0.02 gives ~0.025; ties; small n.
`c.fragile_k` keeps meaning k_safe (persisted as "k").

**H2 - the margin m in the tuner.** `k_eff = m x k_safe` is the threshold the cut selection uses (route_fragile_first and the explore band).
`m` joins the paired tuner's rotation (budget -> mask -> cap -> prefetch -> k), candidates `m x 1.25^d`, range [0.6, 1.5], start 1.0, same
direction/step logic and paired::decide as the other tests. Quality for this test: per arm, the EXPLORE cuts (the first row with ratio in
[k_eff, 8 k_eff), every 16th window, flipped or not) are counted: `auto_xrc[2]`, `auto_xfl[2]`; the candidate arm cannot move when its
explore flip rate's Wilson lower bound exceeds max(control upper bound, base arm's explore upper bound) - i.e. the rows it stopped
re-checking flip above noise - or when quality_veto/cut_veto already block it. Speed alone never lowers m past the veto.
Persist `tune_k_mult` in the .gate file; STRATA_TUNE_K=0 fixes m = 1.0; STRATA_FRAGILE_K_MULT sets the start value only when no state is saved.

**H3 - logs.** The k line adds `base k_safe x.xxx (r* y.yyy, n_t N, f_t F; control upper bound z.zzz)`; the test line `auto k test m a vs b (k_eff x vs y): pairs N,
mean d ..., explore flips f/n vs f/n, -> move|keep (reason)`.

**Tests:** CPU unit tests for k_evidence (cases above), the multiplier candidates (steps, range, persistence round trip), the explore veto;
build; serve tests; import check. Live check by the lead afterwards (frozen-state A/B: k_safe vs old rule vs re-check off, then a day of use).
