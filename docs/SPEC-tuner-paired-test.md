# SPEC: the on-line tuners move only on a real difference (audit items 5 and 6)

*8 Nov 2026. Branch `tuner-paired-test` (from main). Covers the auto budget test (STRATA_ROUTE_BUDGET_AUTO) and the auto mask test
(STRATA_MASK=learn) in src/core/verify.cpp; see docs/AUDIT-functions.md on branch `audit`.*

## What is wrong (code + live log, 8 Nov)
- Both tuners decide with `won = sc >= sb * 1.02 && quality` (verify.cpp ~5245): a point comparison of two tok/s figures with a 2% margin,
  no variance, no sample-size rule. Arms: alternating runs of 8 windows (kAutoRun), 96 (budget) / 384 (mask) windows per arm.
- Live: across tests, (candidate - current) tok/s had sd 1.79 (budget, 35 tests) and 0.91 (mask, 34 tests), means -0.06 and 0.08. At those
  sds a zero true difference "wins" ~35% (budget) / ~25% (mask) of tests. The budget moved 11 times in 35 tests - what pure noise gives;
  the mask alternates between (0.550, 1.0) and (0.575, 1.0). The settings are a random walk, not a learned optimum.

## Objective
A move happens only when the candidate is faster by at least the minimum effect with 95% confidence, and not worse in quality; otherwise
the current setting is kept. Measured: with two identical arms (A/A), the false-move rate is <= 5% of tests.

## Method (one shared decision routine, used by both tuners)
1. **Paired runs.** Keep the existing run interleaving, and pair each candidate run with the adjacent base run (same request when
   possible). Per pair: d_i = ln(rate_cand_i / rate_base_i), rate = committed tokens / wall seconds of that run (the existing per-window
   accounting, summed per run, a cut's exact successor charged as today).
2. **Sequential decision on the mean of d** (Welford mean and variance over the pairs), minimum effect delta = ln(1.03) (setting
   `STRATA_TUNE_MIN_EFFECT`, default 0.03):
   - after at least 12 pairs: **move** when mean - 2.0 * se > delta (one-sided ~97.7%, conservative for the repeated looks);
     **keep** when mean + 2.0 * se < delta;
   - at the cap (budget 160 pairs, mask 240 pairs): **keep**.
   The default outcome is keep.
3. **Quality veto (unchanged in spirit, one denominator):** flips per committed token, Wilson 90% bounds as today; a candidate whose lower
   bound exceeds the base's upper bound, or whose rate is > 1.5x the base's with >= 3 flips, cannot move.
4. **Mask:** drop the "first win, the next test repeats it" rule - the sequential test replaces it. Step sizes and directions as today.
5. **Budget:** the step rule as today, driven by the new move/keep outcome.
6. **Log per test:** `auto budget test X vs Y: pairs N, mean d +0.012 (se 0.008), -> keep|move (reason); flips ...` - so every decision
   can be audited; the same for the mask.

## Tests (no server, no port, no 8095)
1. Unit, decision routine: (a) A/A - 10,000 simulated tests with pair d ~ N(0, sigma) for sigma in {0.02, 0.05, 0.1}: false-move rate
   <= 5%; (b) a true 6% gain at sigma 0.05 moves in >= 80% of tests before the cap; (c) quality veto blocks a faster candidate that flips.
2. The state machines: a scripted sequence of pair results drives budget and mask moves as expected (move on a clear win, keep on noise,
   step growth / shrink unchanged).
3. With STRATA_ROUTE_BUDGET_AUTO off and STRATA_MASK not learn: behaviour identical to today.
4. Build passes; `serve` tests (364) pass; server import check.

## Live check (later, only when 8095 is idle and the user says go)
A day of normal use: moves per test (expect few), the logged mean d and se per test, decode tok/s and flip rate vs today's build.
