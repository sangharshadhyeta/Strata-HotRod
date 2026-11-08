# SPEC: control windows so the fragile re-check and the mask have their baselines (audit items 3 and 4)

*8 Nov 2026. Branch `control-windows` (from main). src/core/verify.cpp, src/kernels/cuda/native_router.cu.*

## What is wrong (code, checked; live log 8 Nov)
- **Fragile re-check noise control never runs** (`control 0 re-checks 0 flips` in all 413 requests). Its selection (verify.cpp ~4882)
  needs an accepted row r >= 1 whose row_shift is exactly 0, i.e. no stand-in moved it in any layer. With STRATA_ROUTE_RESIDENT=1e30 at
  ranks 3-9 in every layer (plus the mask's moves in the same row_shift) such a row practically does not exist.
  Consequence: route_ctrl_bounds with no samples gives ctrl_hi = 1, so `if (target > old && !(ring_lo > kctrl_hi)) target = old;`
  (~5030) blocks every raise: **k can only fall**. The recheck brake's `flip_lo > max(0.2, ctrl_hi)` (~4965) can never fire.
- **Mask learn has no "without" side** (`with a mask move 22/91, without 0/0`). mlr_update (~6639) moves p by evidence only when
  nm >= 16 and fsn >= 16; the only source of fsn is the mask control window, which route_budget_window (~4707) enables only with
  STRATA_MASK_RESIDENT (opt-in, off). Consequence: **the mask's quality brake (p down when mask moves flip more than the baseline) never
  runs**; p is set by the speed test alone.

## Objective
Both baselines get real samples in normal use: >= 1 fragile control re-check and >= 1 mask "without" re-check per ~2,000 decode tokens
on 8095's traffic, at a decode cost <= 1% (measured), so the k raise, the recheck brake and the mask's evidence rule can act.

## Method
1. **Fragile control window (new):** every `STRATA_FRAGILE_CTRL_EVERY` eligible decode windows (default 128; eligible = k set, decode,
   fragile on), a control window runs with **stand-in moves off**: route_resident_k (and the mask kernel) take a device flag, like
   route_mask_control's (native_router.cu ~818-827 / ~972-977), and keep every pick at its router rank; its row_shift stays 0. The existing
   selection then finds r = 1 and the existing settle records exact-vs-exact as the control. The flag is device memory written between
   windows (graph-safe, no host CUDA call inside a window).
2. **Mask control windows without the resident rule:** drop `mask_resident_on()` from the gate in route_budget_window (~4707), keeping
   `mask_learn_active() && decode && mask_control_next()`, so the existing control path (mask moves off, stand-ins on) runs with
   STRATA_MASK=learn alone. Cadence unchanged (kMlControlEvery, halved while fsn < 16).
3. The two control kinds never coincide in one window (a fragile control window takes precedence; the mask control waits for the next).
4. Stats lines: the fragile line already prints `control N re-checks M flips`; the mask line adds `; control windows N` (as with
   RESIDENT). Count the windows of each kind and their tok/s against the others (one line at the request end:
   `control windows: fragile N (x tok/s vs y), mask M (x vs y)`).
5. Settings: STRATA_FRAGILE_CTRL_EVERY=0 turns the fragile control window off; STRATA_MASK_CTRL=0 restores the old mask gate.

## Tests (no server, no port, no 8095)
1. Kernel parity: with the control flag set, route_resident_k and the mask kernel return the router's own picks (no move, row_shift 0)
   on a recorded routing input; with it clear, the same output as today.
2. Unit: the selection picks r = 1 in a control window; the cadence counters; the two kinds never coincide; with the settings at 0,
   behaviour identical to today.
3. route_ctrl_bounds with a filled ring gives a finite ctrl_hi, and a k raise passes when ring_lo > ctrl_hi.
4. Build passes; `serve` tests (364) pass; server import check.

## Live check (later, only when 8095 is idle and the user says go)
Control re-checks per request and their flip rate (the noise floor), the mask's fsn and any p moves by evidence, k over the day,
decode tok/s of control windows vs others and overall vs today's build.
