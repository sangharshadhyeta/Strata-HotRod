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
At a cut's exact successor the learn tap already captures the first stand-in's y_s (cut window) and y_e (exact window, layer l, rank r)
as N-vectors. The logger adds the expert's INPUT x at that layer:
- In the stand-in (cut) window, at the layer l where `lrn_rec_[row]` names its first stand-in, a tiny predicated kernel (after the
  routing kernel of that layer, before the experts) copies the row's layer input (`mixed_` row, N floats) into `lrn_x_` (device,
  allocated only when the logger is on). One launch per layer; nothing when the logger is off (graphs unchanged).
- On the host, at the exact successor's settle (standin_learn_settle, between windows - no CUDA call inside a window, issue #31),
  write one record to `data/subfit-samples-<fingerprint>.bin`: header (magic, N, layers, version); per record: layer u16, e u16, s u16,
  rank u8, x[N] bf16, y_s[N] bf16, y_e[N] bf16. Cap the file at 300 MB (stop writing, log once). Size per record ~15 KB.
- Request-end line: `subfit log: N samples written (file X MB)`.
- Tests: record round trip (CPU); with the switch off nothing is written and no kernel is added.

## Stage 1b - the offline fit tool `tools/subfit_fit` (standalone C++, no server, no GPU needed)
Reads the samples file and the pack (`/mnt/nvme/strata-data/packs/iq3_s`, via FileExpertSource::blob(layer, expert) and the CPU expert
path in include/strata/kernels/cpu/native_expert.hpp: native_quant_act, native_gu_rows, native_quant_h, native_down_rows). For each
sample it computes the substitute's hidden activations h_s(x) (the 512-wide SwiGLU output before the down projection) and reports,
pooled over samples and per layer band (0-11, 12-23, 24-35, 36-47), plus a held-out half (even / odd sample index):
1. cos(y_s, y_e) and the relative error as is (cross-check of the table);
2. **the weight nudge:** ridge regression of y_e on h_s: y_e ~ (W_down_s + dW) h_s, i.e. a linear map M (N x 512), lambda chosen by the
   held-out half; report the relative error on held-out data for full-rank M and for rank 4, 8, 16, 32 (truncated SVD of M - W_down_s);
3. **the better substitute:** for each sample, E_c(x) for up to 64 random resident experts c of that layer (resident = the profile's
   top experts, `data/expert-profile.bin`) plus s itself; the best cosine against y_e over the candidates, and the share of samples
   where the best candidate beats s;
4. the combination: best candidate + the rank-16 nudge fitted for it.
Output: a table (energy of y_e explained, held-out), and a one-line verdict against the go / no-go: stage 2 is built if the held-out
rank-16 nudge (or the best-candidate pick) removes >= 40% of the remaining error energy after the scalar.
Tests: the regression on synthetic data with a known M (recovers it), the SVD truncation, the held-out split, the candidate search.

## Stage 2 (only after the verdict; separate spec)
Per-pair low-rank variants fitted in idle time (pairs chosen by the fit's held-out score), stored as separate deltas (the original
substitute keeps serving its own tokens), applied inside the expert kernel as one rank-r product on the 512-wide hidden state, the
better-substitute table consulted by the stand-in kernel; memory in VRAM for the most-used pairs (~100 KB each), the rest in RAM.

## Hard rules
No server or port access in the build; the logger runs live only when the lead starts it on 8095. Build only via safe_build.sh.
