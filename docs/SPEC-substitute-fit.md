# SPEC: make each substitute match the real expert - measure first (stage 1), then fit (stage 2)

*8 Nov 2026. Branch `substitute-fit` (from recheck-gated 43d803f24). The user's idea: nudge the substitute's weights toward the real
expert so its output matches, and choose better substitutes at the same time. Stage 1 is measurement only: it decides whether stage 2 is
built.*

## Why (measured, 8 Nov)
Stand-in table, 120 pairs with >= 8 samples: median cosine between the stand-in's output and the true expert's output **0.38**; error
relative to the true output median **1.27** as is, **0.99** after the best scalar (the scalar c sits at its 0.5 clip). Counter D: flips
are the sum of many small shifts (flipped share of swaps ~39% at every rank / layer band). The re-check costs ~26% of decode (41.0 vs
30.4 tok/s). The pairs are biased toward re-checked (bad) rows, so the typical pair is probably better.

## Stage 1a - the sample logger (opt-in `STRATA_SUBFIT_LOG=1`, off by default)
Built in `src/core/verify.cpp` (standin_learn_begin_window / standin_learn_cut / standin_learn_settle / subfit_probe_collect), the device
side in `src/kernels/cuda/subfit_capture.cu` (header `include/strata/kernels/subfit_capture.hpp`), the file format in
`src/core/subfit_samples.cpp` (header `include/strata/core/subfit_samples.hpp`), the host call sites in `src/program/generate.cpp`
(path at startup, `subfit_probe_collect()` after each `ver.run`). The logger turns the learner on in observe mode when
`STRATA_STANDIN_LEARN` is unset or 0 (no correction is applied). It needs the same route-resident config as the learner.

Dimensions of this pack (checked against `data/` and the expert source): n_embd N = 2560, expert hidden n_ff = 640, 10 routed experts
per row, 512 experts per layer, 48 layers. (Earlier text said 512 for the hidden width; that was the expert count.)

**Stand-in samples (tag 1).** In the stand-in (cut) window, at the layer `lrn_rec_[row]` names as the row's first stand-in, a predicated
kernel (`subfit_capture_x`, after the route-resident kernel, before the experts; one launch per layer; gate[0] == 1) copies the row's
layer input (the normed `mixed_` row) into `lrn_x_`. `standin_learn_cut` copies it to the host with the y_s it already copies. The record
is written in `standin_learn_settle`, between windows (issue #31: no CUDA call inside a window).

**Probe samples (tag 2) - resident picks at ranks 0-9 (coordinator's addition).** In a probe window (one in `STRATA_SUBFIT_PROBE_EVERY`
decode windows, default 64; set at the window's begin in gate[3]), for row 0 only:
- `subfit_probe_pre` (before `route_prior`/mask/route-resident) records row 0's router picks per layer and clears the flags;
  `clean[layer] = 1` when row 0 had no stand-in at an earlier layer (rec[0] == -1).
- `subfit_probe_post` (after route-resident, before the router mark) sets `flag[layer][k] = 1` when the pick is still the router's
  (not replaced by route-resident or the mask: pre == post), its expert is resident (d_res >= 0) and row 0 is clean; it copies row 0's
  layer input to `x` and the router weights of row 0 to `w`.
- `subfit_probe_y` (after the experts' join, before the combine) copies row 0's K raw expert outputs of the layer (parts_) to `y`.
- `subfit_probe_collect()` (host, after the window): one (layer, rank) is drawn at random (std::mt19937_64, fixed seed) among the
  flagged picks and one record is written. Windows with no flagged pick are counted and write nothing.
A pick that route-resident or the mask replaced is never flagged (pre != post), so the sample is always the router's own expert.

**File** `data/subfit-samples-<model fingerprint>.bin` (beside `standin-tune-<fp>.txt`). Header: magic u32 "STSB", n_embd u32,
n_layers u32, version u32 (1). Each record starts with a tag byte (1 or 2) - the two layouts share one file:
- tag 1: layer u16, e u16, s u16, rank u8, x[N] bf16, y_s[N] bf16, y_e[N] bf16 (~15 KB);
- tag 2: layer u16, e u16, rank u8, w f32, x[N] bf16, y_e[N] bf16 (~10 KB). `e` is the true expert, `w` the router weight of the pick
  (the combine multiplies the raw y_e by it), there is no `s` (the tool chooses the substitute).
The file is capped at 300 MB: past the cap nothing more is written and the cap is logged once.

**Request-end line:** `strata serve: subfit log: N samples written (file X MB), probe windows P, no candidate Q`.

**Tests (CPU, `tests/core/subfit_samples_test.cpp`, target `subfit_samples_test`):** bf16 rounding (relative bound, ties, inf, NaN),
env parsing, a round trip of both tags (exact fields, bf16 values), append to a matching file and refusal of another width, the cap
(one record fits, the next is refused, the file stays within the cap), and a writer that was never opened writes nothing.
Not testable without the GPU: the kernels and their graph behaviour (ready to run on the lead's server, not run here).

## Stage 1b - the offline fit tool `tools/subfit_fit` (standalone C++, no server, no GPU needed)
Reads the samples file and the pack (`/mnt/nvme/strata-data/packs/iq3_s`, `FileExpertSource::open` then `blob(layer, expert)`, and the
CPU expert path in `include/strata/kernels/cpu/native_expert.hpp`: native_quant_act, native_gu_rows (which already outputs the hidden
h = silu(gate.x) * up.x, 640 wide), native_quant_h, native_down_rows). For each sample it computes the substitute's hidden h_s(x) and
its output, then reports pooled over samples and per layer band (0-11, 12-23, 24-35, 36-47), plus a held-out half (even / odd sample
index):
1. cos(y_s, y_e) and the relative error as is (cross-check of the table; y_s from the file is compared with the recomputed one);
2. **the weight nudge:** ridge regression of y_e on h_s: y_e ~ (W_down_s + dW) h_s, i.e. a linear map M (N x 640), lambda chosen by the
   held-out half; the relative error on held-out data for full-rank M and for rank 4, 8, 16, 32 (truncated SVD of M - W_down_s);
3. **the better substitute:** for each sample, E_c(x) for up to 64 random resident experts c of that layer (resident = the profile's
   top experts, `data/expert-profile.bin`) plus s itself; the best cosine against y_e over the candidates, and the share of samples
   where the best candidate beats s;
4. the combination: best candidate + the rank-16 nudge fitted for it.
5. **per rank (tag 2 probes, ranks 0-9 separately, held-out half):** for each probe sample the best resident candidate c != e (up to 64
   random resident experts of that layer plus the profile's top ones) and the relative error of w * E_c(x) against w * y_e, as is, after
   the best scalar, and after a rank-16 ridge nudge fitted on the other half. The router weight w is the record's (the error is reported
   x w as the spec asks: the same error in the output space the combine sees).

Output: a table per band and per rank (energy of y_e explained, held-out), and a one-line verdict against the go / no-go: stage 2 is built
if the held-out rank-16 nudge (or the best-candidate pick) removes >= 40% of the remaining error energy after the scalar.
Tests: the regression on synthetic data with a known M (recovers it), the SVD truncation, the held-out split, the candidate search.

## Stage 2 (only after the verdict; separate spec)
Per-pair low-rank variants fitted in idle time (pairs chosen by the fit's held-out score), stored as separate deltas (the original
substitute keeps serving its own tokens), applied inside the expert kernel as one rank-r product on the 512-wide hidden state, the
better-substitute table consulted by the stand-in kernel; memory in VRAM for the most-used pairs (~100 KB each), the rest in RAM.

## Hard rules
No server or port access in the build; the logger runs live only when the lead starts it on 8095. Build only via safe_build.sh.
