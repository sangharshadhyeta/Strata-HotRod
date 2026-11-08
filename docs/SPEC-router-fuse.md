# SPEC: the router stage at near its plain cost, with every feature in it kept

*8 Nov 2026. Branch `router-fuse` (from combined-nov8, a55452c61). The user's rule: nothing is removed; the stage gets cheap.*

## Measured (8095, STRATA_DECODE_STAGE_TIMERS=1, calibration prompts, 200 windows)
Per decode window (~67 ms summed): **router 12.8 ms (19%)**, cpu-wait 14.2, pcie-experts 9.0, vram-experts 7.8, gdn 7.4, hc-read 4.0,
ffn-read 3.6, attn 3.2, other 2.1, plan-wait 1.3. That is ~0.27 ms per layer for what was one 512 x n_embd GEMV and a top-10.
With STRATA_LOOKAHEAD_GPU=0 decode was +8.7% (one sample) - about half the router stage is the look-ahead node.

## What runs in the stage today (src/core/verify.cpp, between the kStFfn and kStRouter marks, ~2944-3142), per layer, on cs
1. **The look-ahead node** (lookahead_node, on the main stream by default; off-stream was measured slower on 20 Oct because the host
   plan waits for the guesses): for each distance d in {1, 2, 4}: route_pred GEMV (the plain router of layer l+d applied to x_l),
   route_pred_bias_add, la_learn_forward (the low-rank correction B(A x)), native_router_top10_multi, native_route_mask (mask), 
   route_pred_edge_skip, route_pred_tail_read; and la_mtp_forward on the MTP source layer.
2. **The layer's own routing:** moe_route_window (router GEMV + top-10), la_learn_label (copies the logits as the learner's labels),
   native_route_mask (the mask's moves, the events, the learn tap), native_route_resident (the stand-in moves, row_shift, the gaps,
   fragile tops).

## Step 0 - measure each kernel (required before any change)
A micro-benchmark (bench/micro, as the existing ones) that runs each of the kernels above at the live shapes (n = 1..8 rows, NE = 512,
the model's n_embd, K = 10, the learner's ranks 4/8/16, the mask and residency tables filled as on 8095) and prints the time of each,
plus the time of the whole per-layer sequence as one CUDA graph. Output a table; it decides which of steps 1-3 matter most.

## Step 1 - the look-ahead node: three distances in one pass
- **One GEMV for the three distances:** stack the plain routers of layers l+1, l+2, l+4 as one [3*512 x n_embd] matrix (built once at
  load, from the same weights) and compute all three distances' logits in one GEMV that reads x_l once.
- **The learner's correction in the same kernel:** u = A x (28 rows per (layer, dist), already batched) and the B u add fused into the
  GEMV's epilogue (or one small kernel for all three distances).
- **One selection kernel for all three:** bias add, top-10, the mask rule, edge skip and the tail-read mark in one kernel over the 3
  distances x n rows (each block one (dist, row)).
- Same outputs: the route_pred tables (ids, weights, logits) bit-identical where the reductions match, and id-identical (top-10 sets and
  order) on recorded inputs except exact float ties.

## Step 2 - the layer's own routing: one post-routing kernel
- moe_route_window keeps the GEMV; the label copy, the mask move, the stand-in move, row_shift, the gaps / fragile tops, the mask events
  and the learn tap are done by **one kernel** after it (one block per row), reading the logits once from shared memory.
- Same outputs as today's kernel chain, checked on recorded inputs (ids, weights, row_shift, gaps, events, tap values).

## Step 3 - fewer nodes
Kernels that do nothing in a window (e.g. tail read when its mode is 0, the mask when no expert of the layer is masked) are not graph
nodes at all for that configuration (decided at capture from the configuration, never per window).

## Settings
`STRATA_ROUTER_FUSE=0` restores today's kernel chain (both steps); `STRATA_ROUTER_FUSE_LA=0` / `STRATA_ROUTER_FUSE_POST=0` each step.

## Tests
1. Parity on recorded inputs (dump x_l, logits, residency, mask tables of a few live-like windows from the micro-benchmark generator):
   every output of steps 1 and 2 against today's kernels.
2. The micro-benchmark before / after: the per-layer router time.
3. Build, the existing route_* parity tests, serve tests, import check.
4. Live on 8095 (idle only; the user's go is given): calibration prompts at temperature 0 with stage timers: router ms/window before/after,
   tok/s, and the three answers read by hand.
