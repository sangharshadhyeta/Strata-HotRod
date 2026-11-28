# The SYCL engine, option C: shared host code instead of hand copies

Branch `sycl-c` (Strata-HotRod). Goal: Intel GPUs get the engine's current features. Until now `sycl/src` held hand
copies of 76 host files (dpct output plus hand fixes, synced 22 Oct 2026, commit 2eb34c23d), so nothing done in `src/`
since then reached the SYCL build. Option C (chosen 27 Nov) shares the real host files the way upstream's Metal engine
does (`metal/include/metal_compat`): a compat layer provides the CUDA runtime host API over SYCL, the real files compile
unchanged, and the Intel-specific behaviour lives in the real files under `STRATA_GPU_SYCL`.

Nothing here has run on an Intel GPU: this machine has none. Every step is compiled with oneAPI 2026.2 (icpx); running
it is the testers' part (issue #7).

## How the port is put together

- `sycl/include/sycl_compat/`: the CUDA runtime host API over SYCL; the real host files compile against it unchanged.
- Real headers carry the SYCL-only members under `STRATA_GPU_SYCL` (CUDA and HIP see no change).
- `sycl/src/prefill/prefill.cpp` and `sycl/src/program/generate.cpp` are GENERATED: our real file plus numbered Intel
  pieces (`sycl/tools/rebuild_prefill.py`, `rebuild_generate.py`).  Rerun them after our files change; never edit the
  outputs.
- `sycl/src/core/verify.cpp`: the Intel copy with our changes merged in.
- `sycl/src/kernels/**.dp.cpp`: the Intel kernel copies with our `.cu` changes merged in (SYCLomatic 21, then
  `tools/s3_migrate.py`, `s3_merge2.py`, and by hand where they collided).

## State (28 Nov)

Every file of the SYCL build passes the compiler front end (oneAPI 2026.2).  Link: see the newest commit message.

In the port: the engine's host features (stand-ins, look-ahead learner, K/V streaming, masking, RAM tier and
read-ahead, the stand-in / sub-fit / latent / layer learners, MTP, prompt cache), `--stream-experts` with the pinned
host mirror (`STRATA_MIRROR_MIB`, `STRATA_LEND_MIRROR`), split headroom, refuse_exit, `STRATA_USAGE_DUMP`,
`STRATA_DBG_DRAFT`, the grouped-XMX prompt paths (`STRATA_PF_XMX=1/2`), the short first chunk
(`STRATA_PREFILL_FIRST`, default 256 when the cache holds every expert), the setup timers.

Opt-in on Intel until measured there: `STRATA_SYCL_SPLIT_OVERLAP=1` (the overlapped layer split; refused on Arc
A-series, and its hand-off wait has no timeout), `STRATA_MMVQ_IL=1`, `STRATA_SELECT_SIMT=1|2`, `STRATA_EXPDEC=1`.

Not in the port (the default path is used instead):

- the Q8 hyper-connection read (`STRATA_HC_Q8`, which the engine switches on by itself for a Q8_0 file, and
  `STRATA_HC_REQ8`): both stay on the BF16 read, with a message
- the prompt stager's batched source reads (`copy_blobs`) and "a failed read stops the request, not the engine": the
  SYCL stager (one job at a time) is kept
- NVIDIA hardware paths: the TF32 selection scorer, the mma prompt attention (the XMX port runs instead), PDL, the
  Turing / Pascal / Volta kernels, `STRATA_HC_UPMIX`, the fused GR read's QFUSE
- GPU-clock timings: Intel has no device clock here, so the stage stamps use a host clock and the look-ahead learner's
  own kernel timing reads 0

Known, not fixed (in the Intel copy before this work, read not run): `fused_gr_read_multi`'s chunked fallback (only
with `STRATA_GR_DOWN_SLICED=0` and `STRATA_GR_DOWN_DIRECT=0` on a card whose local memory holds fewer than the window's
tokens) writes its partials by the chunk's token index and reads them by the window's.

Nothing has run on an Intel GPU.  Correct answers and speed are the testers' to establish (issue #7).
