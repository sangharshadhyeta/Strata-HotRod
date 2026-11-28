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

## S3: the kernels (from tmp/sycl_kernel_survey: our .cu changes since the copies)

| .cu | lines changed | new kernels the SYCL copy cannot have |
|---|---|---|
| `kernels/cuda/native_router.cu` | 2,438 | the look-ahead learner (`la_b_*`, `la_fwd_*`, `la_mtp_pub_k`), MTP zero-copy (`mtp_slot_remap_k`, `mtp_zc_mark_k`), `pred_bias_add_k`, `pred_tail_read_k`, `route_mask_k`, `route_post_k`, `route_pred_sel_k`, `route_tally_k`, `router_gemv_top10_fused_k` |
| `kernels/cuda/verify_kernels.cu` | 119 | - |
| `kernels/cuda/fused_gr.cu` | 92 | - |
| `kernels/cuda/native_bf16.cu` | 79 | `bf16_f32_mmvf_multi3_kernel` |
| `kernels/cuda/s2_expert_grouped.cu` | 49 | `group_ptr_kernel` |
| `kernels/cuda/iq_kernels.cu` | 49 | `nvfp4_dequant_gu_kernel`, `nvfp4_dequant_sub_kernel` |
| `prefill/kernels.cu` | 38 | `standin_agree_mark_k`, `standin_agree_restore_k` |
| `kernels/cuda/native_moe.cu`, `kv_stream.cu`, `sampler.cu`, `shared_expert.cu`, `elementwise.cu`, `ple.cu`, `gdn.cu` | 2-29 each | - (signature changes: e.g. `native_moe_combine` takes a `MoeLearnTap`) |
| new `.cu` with no copy | | `core/remote_expert_opt.cu`, `kernels/cuda/subfit_apply.cu`, `subfit_capture.cu`, `latent.cu` (+ parity tests) |

Two ways: re-migrate the changed `.cu` with SYCLomatic and 3-way merge onto the copies (how `tools/merge_upstream.py`
keeps them; SYCLomatic 20260928 is unpacked in /home/intel/syclomatic, its use needs the maintainer's go), or port by
hand.  Kernels with no SYCL version yet can start as stubs that report "unsupported" so the features they serve stay off.

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

## State 28 Nov evening: everything compiles; the link misses 170 symbols (sycl/S3-undefined.txt)

`prefill.cpp` and `generate.cpp` are now built the option C way: our real file plus the Intel pieces, by
`sycl/tools/rebuild_prefill.py` and `sycl/tools/rebuild_generate.py` (rerun them after our files change).  generate's
pieces: `--stream-experts` (our GGUF-mapped source; the SYCL copy's pinned host mirror is not carried), the xe-driver
PCIe rule (no pinned mirror, so the share defaults to 0 on xe), the non-resident share for the prompt path, leaving
without unwinding at exit, graph warm-up at start, `logprobs=K`.  Not carried yet: the pinned host mirror
(STRATA_MIRROR_MIB / LEND_MIRROR), the split-headroom pick (STRATA_SPLIT_HEADROOM), refuse_exit at start-up refusals,
the usage dump, STRATA_DBG_DRAFT.

Next:
1. verify.cpp the same way (ours + Intel pieces): the merged one compiles but lost Verifier methods (set_logit_bias,
   evict_batch_graph, nan_guard_on, note_nonfinite ...).  Its pieces: stepped windows (capture_segments, step hooks),
   eager replay, STRATA_WINDOW_SYNC, warm(), the debug ladders, the 256-expert router, chain_overlap off.
2. S3 with SYCLomatic (/home/intel/syclomatic/x/bin/dpct, 21.0.0): re-migrate the changed .cu files (tools/migrate.sh)
   and 3-way merge onto the .dp.cpp copies (tools/merge_upstream.py), then the new .cu files.

## S3 in progress (28 Nov night): kernel merge done, 222 conflicts to resolve by hand

SYCLomatic 21 (/home/intel/syclomatic) re-migrated the 36 changed/new .cu at the base and at HEAD (CUDA 12.8 headers in
/home/intel/cuda-headers); both normalised (DPCT numbers, fixups.py, hashes mapped to the copies'); merged onto the
copies (tools/s3_merge2.py, outputs in /home/Projects/sycl-s3).  Left (`git grep -c '<<<<<<< sycl' sycl/src`):
fused_gr 46, iq_kernels 33, native_mmvq 31, qsa_decode_attn 22, prefill/kernels 13, verify_kernels 12, native_moe 8,
qsa_select 8, moe_fused_iq 8, native_bf16 6, s2_expert_grouped 6, gemm 5, sampler 4, shared_expert 4, native_router 3,
elementwise 3, ple 3, moe_fused 3, dequant_bf16 2, qsa_prompt_attn 2.  They are real: the SYCL side holds Intel fixes
(e.g. native_router: route_multi hung an A750, so one block per token).  Then build, link, fix what the link names.
