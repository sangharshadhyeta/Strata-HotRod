# The SYCL engine, option C: shared host code instead of hand copies

Branch `sycl-c` (Strata-HotRod). Goal: Intel GPUs get the engine's current features. Until now `sycl/src` held hand
copies of 76 host files (dpct output plus hand fixes, synced 22 Oct 2026, commit 2eb34c23d), so nothing done in `src/`
since then reached the SYCL build. Option C (chosen 27 Nov) shares the real host files the way upstream's Metal engine
does (`metal/include/metal_compat`): a compat layer provides the CUDA runtime host API over SYCL, the real files compile
unchanged, and the Intel-specific behaviour lives in the real files under `STRATA_GPU_SYCL`.

Nothing here has run on an Intel GPU: this machine has none. Every step is compiled with oneAPI 2026.2 (icpx); running
it is the testers' part (issue #7).

## Steps

| Step | What | State |
|---|---|---|
| S0 | baseline: today's `sycl/` tree compiled as it is | done 28 Nov: 114 files compile; the link misses 5 kernels (`native_moe_combine`, `_multi`, `_multi_hits_gated`, `moe_combine`, `doorbell_publish_res`) whose signatures changed since 22 Oct |
| S1 | `sycl/include/sycl_compat/` (cuda_runtime.h, cuda_runtime_api.h, cuda_fp16.h) | done: all 63 real host files that have a copy pass `icpx -fsyntax-only` against it |
| S2a | the small copies' hand changes into the real files | done: router 256 experts (SYCL), 64 read threads (SYCL), `STRATA_PLE_TRACE` (all); the other marks were dpct-migration fixes the compat layer makes unnecessary |
| S2b | the Intel design of `verify`, `generate`, `prefill` (+ `verify.hpp`, `session.hpp`, `mtp.hpp`) | in progress: these six stay SYCL copies for now (their Intel design is ~1,400 hand lines in `verify` alone, tested on Arc; folding it blind into the real files risks breaking it), and our changes are 3-way merged INTO them (base: `fde1d795e`, the upstream source they were migrated from). The headers are merged; of the .cpp conflicts 159 were dpct-only on the SYCL side (taken from ours, each logged in `MERGE-auto-resolved.txt`), 88 are left by hand (`git grep '<<<<<<< sycl'`): verify 24, generate 24, prefill 40 - mostly the window launch, where the SYCL stepped/eager launch meets our split overlap and learner hooks |
| S2c | `sycl/CMakeLists.txt`: compat first on the include path, `STRATA_GPU_SYCL`, the host copies and copied headers removed | after S2b (the copies share class headers with the rest, so they go together) |
| S3 | kernels: SYCL versions of what the features added since 22 Oct call (23 new kernels in `native_router.cu` - look-ahead learner, stand-ins, MTP zero-copy, route mask; plus smaller ones in 7 files; 4 `.cu` with no copy) | after S2 |
| S4 | link, then hand the branch to an Arc tester | |

## What the S2b merge gave up on the SYCL build (to bring back)

- `prefill`: the short first chunk (`STRATA_PREFILL_FIRST`, the GPU starting while the rest of the prompt's PLE rows
  are read) and the setup timers (`lap`, `STRATA_PREFILL_TIMING`): they lived in the chunk loop our side rebuilt
  (multi-job batches, setup/layer phases).
- `prefill`: the stager keeps the SYCL design (one job at a time, GGUF reads by the stager, sequence-numbered copies);
  our batched source reads (`copy_blobs`) and the "a failed read stops the request, not the engine" change are not in it.
- `verify`: the overlapped layer split (a split's next stage prelaunched): `chain_overlap()` is false (no HostGraphLaunch).
- The CPU share of the prompt path (`STRATA_PREFILL_CPU_SHARE`) is new to the SYCL build (the copy predated it).

## The Intel design that has to move (S2b)

From the copies (`grep "SYCL port"`, and the switches only the copies read):

- `verify`: stepped windows for Arc A-series (`STRATA_VERIFY_STEPPED`: one graph per segment, CPU experts served at queue
  boundaries, because DG2 has no host-visible USM atomics), eager replay (`STRATA_VERIFY_EAGER`), the per-generation
  drain before a window (`STRATA_WINDOW_SYNC`, Alchemist), graph warm-up at start (`warm`), `STRATA_VERIFY_LEAN / _NO_HOST
  / _PIPE / _SPLIT`, `STRATA_RING_TIMEOUT_S`, bounded doorbell spins (`sycl_doorbell.hpp`)
- `prefill`: the stream-all walk fix, jobs that name a GGUF expert, sequence-numbered uploads, the short first chunk,
  `STRATA_PREFILL_STREAM_ALL / _FIRST / _SYNC`, `STRATA_GGUF_PLAN_READ`, `STRATA_PF_DQ_FUSED`, `STRATA_PF_XMX_ROWS`
- `generate`: experts read from the GGUF on demand (`--stream-experts`), the pinned host mirror the GPU reads
  (`STRATA_MIRROR_MIB`, `STRATA_LEND_MIRROR`), the xe-driver page-fault rule, borrow chosen by context, logprobs,
  leaving without unwinding at exit, graph capture at start, `STRATA_SPLIT_HEADROOM(_TOL)`, `STRATA_FILL_SERIAL`, ...

Fixes of dpct's own output (lost `cudaStreamQuery` answers, `DPCT_CHECK_ERROR(ext_oneapi_empty())` always 0) are not
carried: the real code calls the CUDA API and the compat layer answers it.
