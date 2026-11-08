# SPEC: stand-in learning that gets enough samples (Qwen engine)

*8 Nov 2026. Branch `standin-samples` (from main). Covers STRATA_STANDIN_LEARN and STRATA_STANDIN_BALANCE (src/core/verify.cpp,
include/strata/kernels/learn_tap.hpp, the combine kernels in src/kernels/cuda/native_moe.cu and shared_expert.cu).*

## What is wrong (measured on the live table, data/standin-learn-5400e1aece241c0f.bin, 8 Nov)
- 5,566 pairs (layer, true expert e, stand-in s), 10,363 samples: **12 pairs ready** (kLrnReady = 32), 4,174 pairs seen once.
- **85% of the samples are layer 0** (8,832); layers 17-47 have 1-2 samples each, most none. Cause: a cut gives one sample, the cut
  row's *first* stand-in (the only one whose input equals the exact path's), and the first stand-in is nearly always layer 0.
- So STRATA_STANDIN_BALANCE ran 4,377 rounds and never kept or gave up an expert (a layer needs >= 10 ranked experts).
- Pooling the sums predicts a pair well (leave-one-out over the 143 pairs with >= 8 samples): |c - group c| 0.037 by (layer, e),
  0.055 by (layer, s), 0.057 by layer, against |c - 1| 0.44 for the untrained default.

## Fix A - pooled estimates (host only, verify.cpp)
Keep the per-pair sums as today, and also the same decayed sums (num, den, ee, n) per (layer, e) and per layer, updated with every
sample. A pair's estimate uses **shrunk sums**: `S_pair + K * S_parent / n_parent` with K = 16 pseudo-samples, where the parent is
(layer, e) when it has >= 32 samples, else the layer when it has >= 32, else none (the pair alone, as today). From the shrunk sums, c and
err as today (c clipped [0.5, 2]). A pair is **ready** when its parent exists or its own n >= 32.
- **The risk table** (`standin_learn_pe_layer`) and **the balance** (`standin_balance_commit`: "best err of the expert's pairs") use the
  pooled err; an expert e with a (layer, e) group of >= 32 samples is ranked even with no single ready pair.
- **The corrections table** (lrn_c_, the scale applied to a stand-in's output): the pooled c only with `STRATA_STANDIN_POOL_APPLY=1`
  (opt-in until an A/B is read by hand); otherwise today's rule (own n >= 32).
- The parent sums are rebuilt from the pairs at load (no file format change; an LRU-evicted pair's samples stay in its parents).
- Stats line: `stand-in learn: P pairs ready (Q own), G experts pooled, ...`.

## Fix B - a clean sample from every stand-in of the cut row, at every layer
Today: rec[row] keeps the row's first stand-in only; the exact successor captures y_e at that one (layer, rank).
- **Record up to M = 8 stand-ins per row** (layer, rank, e, s), the first M in layer order (rec becomes [rows][M][4]).
- **The exact successor captures, for each recorded (layer, rank) of the cut row:** y_e (the rank's raw part, as now) **and x_e**, the
  expert input at that layer for row 0 (the same vector the expert's gate/up read). Buffers ye[M][N], xe[M][N]. Same predicated stores
  inside launches that already run (graph shape unchanged); the gate holds the M (layer, rank) targets in device memory.
- **After the window (host, outside any window - issue #31):** the first record keeps today's sample (y_s from the cut window equals
  s(x_e) there). For each other record, a host worker thread computes **y_s = s(x_e) on the CPU** with the engine's existing CPU expert
  kernels (src/kernels/cpu, the CPU path's dequant + gate/up/down for this pack type), reading s from the pack as the CPU path does, and
  adds the (y_s, y_e) sample to the pair (and its parents). The worker has a bounded queue; when full, the sample is dropped and counted.
  It never blocks decoding, never issues a CUDA call inside a window, and runs at the lowest priority.
- With STRATA_STANDIN_LEARN unset: no buffers, no stores, graphs identical to today's.

## Tests (no server, no port, no 8095)
1. **Pooled estimate:** unit test of the shrinkage (a pair with 1 sample and a parent with 100 takes the parent's c; with 1,000 own
   samples, its own), the readiness rule, and the parent rebuild at load. Old table files load unchanged.
2. **Balance acts:** fed the live table copy's pairs (read-only, copied to a temp dir), the balance ranks >= 10 experts in the layers
   that have data and keeps / gives up the top / bottom decile.
3. **CPU parity:** s(x) on the CPU equals the GPU's part for the same expert and input (the existing native_expert_parity pattern),
   within the pack type's parity tolerance.
4. **Capture:** with a test window, the M recorded (layer, rank) targets' y_e and x_e equal the exact path's values.
5. The build and the existing parity tests pass; `serve` tests pass (364); server import check.

## Live check (later, only when 8095 is idle and the user says go)
Samples per request and their spread over layers (target: every layer with stand-ins gets samples), pairs ready, balance decisions
per round, decode tok/s and the flip rate against today's build on the same prompts.
