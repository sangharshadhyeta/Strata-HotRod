# SPEC: the learned look-ahead and the need-only read-ahead cost nothing when they have nothing to gain

*8 Nov 2026. Branch `adaptive-la` (from combined-nov8, a55452c61). The user's rule: keep every feature, remove its cost.*

## Measured (8095, calibration prompts, temperature 0, one sample each, 96-97% hit rate)
All on 32.3 tok/s (72.5 ms/window); STRATA_LA_LEARN=0 34.0 (+5.5%, 62.5 ms/window); STRATA_LA_TAIL_READ=0 34.1 (+5.8%). Both won when
they were made defaults (24 Oct, ~88% hit rate, long sessions): their value is the SSD stalls they save, which a 96% hit rate leaves
little of, while their cost runs every window. Single samples with different answer lengths: treat as hints.

## Change 1 - decoding first on the GPU (stream priority)
The learner's training batch runs on la_cs_ (verify.cpp ~1893, make_stream), created at the default priority - the same as the decode
streams - so when both have blocks queued the SMs are shared.
- Create the decode stream(s) of the verify window (cs_, and the shared-expert stream sh_cs_) at the device's **greatest** priority and
  la_cs_ at the **least** (cudaDeviceGetStreamPriorityRange). The learner then runs on SMs the window leaves idle; its results are the
  same (the batch still waits for its window's event and the next window still waits for the batch's done event as today).
- Note the existing STRATA_PIPED_CONCURRENT_PRIO path (verify.cpp ~1845) and keep its behaviour.
- `STRATA_LA_PRIO=0` restores today's priorities.

## Change 2 - the training batch backs off while nothing changes
Today a frozen (layer, dist) trains its leader every 4th window (kLaLeaderEvery); a live one trains every window.
- A device-side cadence per (layer, dist): while frozen and its recent leader hit rate stays within its Wilson interval of the long-run
  rate, the leader trains every 4 -> 8 -> 16 windows (doubling after each 2,000 picks without a change, at most 16); on a change, a
  reset, or a recent hit rate below the interval, back to every window (live) / 4 (frozen) at once.
- When no (layer, dist) trains in a window, the batch is not launched at all (one host-side check of a mapped counter the decide kernel
  writes; no launch = no cost). The forward (the predictions) runs every window as today.
- `STRATA_LA_CADENCE=0` restores today's cadence.
- Stats line: `look-ahead learner: trained in N of M windows (cadence median 8)`.

## Change 3 - the need-only read-ahead: measure, then adapt
Its reads already pass a learned gate (expert_source.cpp ~5948, ra_admit: a guess's position must use-rate above the drive's load).
- Add a host timer (steady_clock, summed per request) around the decode guess dispatch (expert_source.cpp ~5918-5977) and print
  `read-ahead dispatch: X ms per request (Y us per window), Z guesses considered, W queued`.
- If that host time is > 2% of decode time: when the last 64 windows queued no tail guess, consider ranks < lo only (gpu_pred_ranks)
  until a miss at rank >= lo is seen again, then return to all 10 ranks. Otherwise leave it unchanged and report the measurement.

## Tests
1. Unit (CPU): the cadence rule (scripted hit-rate sequences: back-off doubling, cap 16, immediate return on drop/change/reset); the
   "no trainer this window -> no launch" check.
2. Parity: with STRATA_LA_PRIO=0 and STRATA_LA_CADENCE=0 the learner's tables after a fixed synthetic sequence equal today's; with the
   cadence on, the predictions (forward) of a frozen leader are unchanged between trainings.
3. Build, serve tests, import check.
4. Live on 8095 (idle only; the user's go is given): calibration prompts and the long-context round, all-on new vs STRATA_LA_LEARN=0 vs
   today's all-on; tok/s, the learner's leader hit rates (must not fall), trained-window share.
