# Inkling 2 on HotRod - the port plan

Started 22 Nov 2026. Goal: Thinking Machines' Inkling-2-Mini (26.4 B, 3.4 B active, a looped MoE) and, once that runs,
Inkling-2-Small (594 B) on this engine's tiers (VRAM / RAM / NVMe), with the stand-ins, look-ahead and learners the Qwen
path has. Inkling is **not a flag**: the engine's Qwen path is compiled around qwen4exp's geometry, so this is a new model
path, built beside it (as Maya's `maya-ik2` and upstream Strata's `strata-hy4` are separate engines).

## What is being ported (from the GGUF's own metadata, read 22 Nov)

`general.architecture = inkling2`, unsloth `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17,629,759,456 bytes, sha256 `e62d463c...de15`).

| | |
| --- | --- |
| Visits | `block_count` 48 = 24 stored layers; layers 6..17 (`loop.core_start/end` 6/18) run `num_loops` 3 times, with control gates (`loop_{attn,k,v,mlp}_{down,forget,gain}`) and "messages" from the pass before |
| Width | `embedding_length` 2304, 24 query heads / 8 KV heads of 128, `feed_forward_length` 1152 |
| MoE | `expert_count` 128, `expert_used_count` 8, no shared expert, sigmoid router with a selection bias, weights normalized x 8 (`expert_weights_scale`) |
| Activation | SwiGLU with softcaps: `4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)` (`situ_beta` 4, `situ_linear_beta` 25) |
| Attention | no RoPE; relative bias from 16 per-head states over `relative_extent` 1024; a sink logit per head and token; L2 clips on q, k (`qk_clip` 22.6); weightless RMS on v; sigmoid output gate; sliding window 1,024 in a 5 : 1 pattern (every 6th layer full), full layers add log scaling (`log_scale_alpha` 0.1, floor 8192) |
| Short convs | 4 taps (`shortconv.l_cache`) on k and v (a full layer's dynamic), after attention and after the MoE |
| Attention residuals | blocks of 6 layers (`attn_res.block_size`): learned queries attend over the block states and the stream |
| Head | logits softcapped at 30 (`final_logit_softcapping`), 200,058 real rows of 201,024 |
| Experts in the file | NVFP4 in layers 0-19 (a per-expert `.scale` tensor, F32, beside each), Q8_0 elsewhere; dense in Q8_0 / BF16 |

## Oracle

Maya's `ref/inkling2.py` (MIT, written by Maya's author against transformers' `inkling2` / `inkling2_loop`) is a float64
forward pass over any `inkling2` GGUF that records each intermediate under a seam name (`blk.V.k_conv`, `.q`, `.k`, `.v`,
`.r`, `.attn_gated`, `.attn_out`, `.picks`, `.ffn_out`, `.out`, `logprobs`). Its own check against transformers on the real
Inkling-2-Small is max |dlogp| 4.7e-5. Every op of ours is compared with its seam before the next is built. We use it as a
test oracle and a specification; the engine code is written here, in this engine's conventions.

## Milestones (each one tested before the next; no big-bang)

- **M0 oracle:** run the reference on our GGUF for a fixed prompt, keep the seam dumps.
- **M1 file + config:** an `inkling2` reader on top of our `gguf_reader` and expert source: metadata, tensor table, NVFP4 / Q8_0 /
  BF16, the per-expert `.scale`; a guard like the qwen4exp one.
- **M2 ops, one by one, each against its seam:** short conv (static, dynamic), attention (relative bias, sink, clips, log scaling,
  gate), attention residuals, router + experts (NVFP4 decode), head.
- **M3 forward:** the 48-visit loop, decode and prompt, sliding / full KV, conv state, per-position block states; end-to-end
  logprobs against the reference.
- **M4 engine:** the expert tiers, stand-ins, look-ahead, serve on 8095, speed against Maya's `maya-ik2` (32.4 tok/s on this
  12 GB card, all experts in RAM), batching 2x / 4x; then Inkling-2-Small from the NVMe.

## Status (22 Nov 2026)

- **M0 done:** oracle dumps for a 3-token and a 48-token prompt (578 seams each).
- **M1 done:** `include/strata/artifact/inkling2.hpp` reads and validates the file (734 tensors, config cross-checks).
- **M2 done, and the looped forward of M3 in plain C++ (CPU, teacher-forced):** `include/strata/inkling2/ops_cpu.hpp` +
  `tests/core/inkling2_ops_test.cpp`. All 48 visits of the 48-token oracle match the reference's seams within 9e-7 (relative to each
  seam's maximum), every expert pick (18,432) is identical, and the log-probabilities differ by at most 7.6e-6 over 48 x 200,058
  values with the same top-1 at every position. It takes ~2 minutes on one CPU thread: an oracle, not an engine.
- **The 48-token oracle cannot exercise** the full layers' log scaling (tau is 1.0 below 8,192 positions), the 1,024-position window or the
  relative extent past 48. **Covered 22 Nov by a 92-token oracle with shrunken values** (the reference run with window 32, full extent 48 and
  log-scaling floor 16; the C++ side applies the same values through `STRATA_INKLING2_WINDOW` / `_EXTENT` / `_FLOOR`): all 48 visits within
  1.13e-6, all picks identical, max |dlogp| 7.6e-6 over 92 x 200,058, top-1 the same at 92 of 92 positions. That tests the window mask, the
  extent cutoff and tau up to 1.17 on the same code path the real values use; a run at the real 1,024 / 8,192 stays out of reach of a float64
  CPU reference, so the real values are covered by this equivalence, not by a direct run.
- **Next:** (a) a long-context oracle; (b) incremental decode: KV for the 8 full-attention layers' visits and the sliding window, the conv
  states (3 rows per conv), per-position block states; (c) the CUDA kernels, each checked against `ops_cpu`; (d) the expert tiers.

## Same concepts as Edge0 and Qwen (decided 22 Nov, on the user's instruction: no bare port)

The engine path is `src/inkling2/`, beside `src/qwen35/` (Qwen3.6 got its own path the same way) and on top of the shared `core/`: of
`include/strata/core`, `expert_source.hpp` (1,580 lines, no Qwen reference), `standin_pool.hpp`, `recheck_gate.hpp`, `pinned*.hpp`,
`expert_cache.hpp` and the learners are model-agnostic (a few `512` expert-count limits, which Inkling's 128 fits under); `layer.hpp`
and `geometry.hpp` are Qwen's forward and shapes and are not reused. What carries over, and what is different here:

| Concept (Qwen / Edge0) | For Inkling |
| --- | --- |
| VRAM expert cache (adaptive slots), RAM tier (`--resident-budget-gib auto`), O_DIRECT NVMe tier, pinned segments with the computed reserve | Experts are keyed by (stored layer, expert): 24 x 128 = 3,072, 4.5 MB each in NVFP4 (layers 0-19: gate_up 2.99 MB + down 1.49 MB), 8.5 MB in Q8_0 (20-23). Mini's 15.8 GB of experts fit this box's RAM tier whole; Inkling-2-Small (594 B) goes to the NVMe tier |
| Stand-ins / residency-biased routing (ranks 3-9, edge 1 on Qwen: top-10) | Top-8 of 128 with a sigmoid router whose picked scores are renormalized x8: the rank policy is re-derived, not copied, and quality is judged by reading answers and by log-prob drift against the oracle (no numeric verdict alone) |
| Learned look-ahead, router prediction, re-check gate | The loop gives a stronger predictor than anything on Qwen: see below |
| Layer-major prompt walk (experts read once per chunk) | A chunk runs its 48 visits in order; the core's three passes need the same layer's experts three times, 12 visits apart, so the passes are where a resident cache pays |
| KV streaming (K/V in pinned RAM) | The KV is small (sliding layers keep 1,024 positions; 8 full layers ~32 KB / token), so 256K context is 4-8 GB: streaming matters only on small cards |
| MTP / drafting | Inkling's drafter is a separate 5-layer DSpark-style head (`mtp.safetensors`), not in the GGUF: later |
| Adaptive swaps, PCIe model, `--pcie-frac`, learners that start with the engine | Unchanged in principle; every learner starts from this engine's own measurements |

**Measured on the 48-token oracle (22 Nov; one text, so a signal, not a tuning number):** for the same token, a core layer's 8 picks in
pass 2 share 4.9 experts with its pass-1 picks (pass 2 vs 3: 4.4; 1 vs 3: 3.65), against 0.5 for chance and 3.4 for the previous token at
the same visit. A pass-2 or pass-3 pick is already in the same token's pass-1 set of that layer 54% of the time. So the look-ahead's best
feature for the looped core is the token's own previous pass, ahead of anything it uses on Qwen. Layer 17, the core's last, is the exception
(2.9 / 1.9). Over the 48 tokens a core layer touches 58 distinct experts in pass 1 and 77 across the three passes (of 128).

## Reuse audit - what the engine already has (read from the tree, 22 Nov; the rule is: build only what is not here)

What was written so far is **test-only**: `inkling2.hpp` (the file reader, new: no `inkling2` existed), `ops_cpu.hpp` (a plain-float CPU
forward used as an oracle for the GPU path; it replaces nothing and ships in no engine path) and the tests. One small duplication, on
purpose and flagged: `ops_cpu.hpp` has its own scalar NVFP4 decode, written independently from gguf-py so the test does not trust the code
it is meant to check; the engine already has the real ones (`src/kernels/cpu/nvfp4_avx2.cpp`, `nvfp4_avx512.cpp`, `native_expert.cpp`, the
CUDA ones in `iq_kernels.cu`; the ue4m3 table there is the same convention). The engine path uses those.

| Inkling op | Already in the engine | New |
| --- | --- | --- |
| Static short convs (k, v, post-attention, post-MoE; the sliding layers' k / v): 4 taps, 3-row state | `gdn_conv_kernel` / `gdn_conv_step` (`src/kernels/cuda/gdn.cu`): `out[c] = sum state[i] w[i] + x w[last]`, weights `[C][taps]`, the state slides; our folded conv weights are that layout | the one-time folding of the conv scale and residual into the weights (done in `ops_cpu`, to move to load time) |
| Full layers' dynamic conv taps | - | a small kernel: static taps + per-token `dyn[t][head*4 + lag]` |
| Dense projections (Q8_0 / BF16): q, k, v, gate, rel, sink, output, the head, the embedding | `native_mmvq.cu` (Q8_0, arbitrary n_in / n_out, packed and rows1 variants); the prompt GEMMs | BF16 for the small sink / conv-proj tensors if not covered |
| RMSNorm (weighted, weightless), L2 clip | `elementwise.cu` norm kernels | the L2 clip with a per-channel scale |
| KV cache, **KV streaming** (K/V in pinned RAM, page pool, q8 / q4 KV) | `kv_stream.cu`, `kv_q8.cu`, `kv_q4.cu`: the pinned host copy, the page pool, the q8 / q4 KV formats, and the **ring** (static table `block -> block % n_slots`, the MTP drafter's window) are parameterised by `n_head_kv`, `head_dim`, `page_size`. **Corrected 22 Nov:** the streaming proper (`kv_stream_resolve`) makes resident the blocks a QSA *selection* names, and Inkling's full-attention layers are dense (every position, every token), so selection-driven streaming does not apply to them as first written here | The KV is **per visit**, not per stored layer (the looped core's K/V differ on each pass): 40 sliding visits (a 1,024-position ring: the existing ring design fits) and 8 full visits (dense: resident in VRAM, q8 KV from the existing formats; ~32 KB / token in FP16 over the 8 visits, so 32K context is ~0.5 GB in q8, 256K ~4 GB). Context beyond the VRAM is an open design question (dense attention over streamed KV is PCIe-bound), not a reuse |
| Attention | `native_flash_attn.cu` is pinned to scale 1/16, "no ALiBi / softcap / sink"; `qsa_*` is Qwen's sparse indexer | the kernel: relative bias (16 per-head states by distance), a sink logit, sliding window, sigmoid output gate, head_dim 128 |
| Router | `native_router.cu`: a pinned contract of 512 experts, top-10, **softmax** | sigmoid + selection bias, top-8 of 128, normalized x8 (adapt the kernel's structure) |
| Expert compute | NVFP4 and Q8_0 expert kernels (CPU and CUDA), the grouped kernels, the per-hit scratch | `s2_expert_grouped.cu` is compiled for Qwen's `H=2560, FF=640` and 2-bit rows (a `static_assert` against `kCompiledGeometry`): the Inkling shapes (2304 / 1152, NVFP4 / Q8_0) need a parameterization pass; the SITU activation (softcaps 4 / 25) and the per-expert F32 scale are new |
| Tiers: VRAM cache, RAM tier, O_DIRECT NVMe tier, pinned segments, ring, swaps, learners, stand-ins | `core/expert_source.hpp` and friends, as above | keyed by (stored layer, expert); the loop-aware prefetch |
| Attention residual over block states; the loop control gate and messages | - | small kernels (a softmax over <= 9 normed states; a rank-64 gate) |
| Sampler, serve path, conversation cache, MTP plumbing | `sampler.cu`, the serve code in `src/program`, `core/conversation_*`, `core/mtp.hpp` | Inkling's chat template and tokenizer (the o200k-style `inkling2` pre-tokenizer) |

**Reuse checked on the GPU, with the engine's own kernels (22 Nov):**

- `gdn_conv_step` runs Inkling's static short convs unchanged: layer 0's k (1,024 ch), v (1,024) and post-attention (2,304) convs, token by token
  with the sliding state, match the CPU oracle at 4.5e-8 / 7.2e-8 / 5.0e-8 and the reference seams at 9.0e-8 / 7.2e-8 / 1.0e-7
  (`src/kernels/inkling2_conv_parity.cpp`).
- `iq_mmvq(40, ...)`, the NVFP4 kernel of Edge0, runs Inkling's NVFP4 experts unchanged (same block layout, same row sizes as the reader):
  a real expert of layer 0 for four tokens, gate / up rows through the kernel, the per-expert scale applied, the SITU activation on the host (the
  one new op), the down rows through the kernel again. Against the CPU oracle's exact-dequant double dots: gate and up within 0.5% of the
  vector's maximum, h within 0.7%, **the expert's output within 1.1-1.5%**. That is the q8_1 activation quantization every expert on this engine
  goes through (it is the same path Qwen uses), not float rounding. It is a per-expert figure; what it does to the answer must be
  measured end to end (log-prob drift and read answers) once the GPU forward exists, not assumed (`src/kernels/inkling2_nvfp4_parity.cpp`).
- Dense Q8_0 (`native_mmvq`), the KV streaming and the router adaptation are next on the list.

So the GPU engine is: the existing tiers + KV streaming + Q8_0 dense + the conv kernel + NVFP4 / Q8_0 expert kernels (parameterized) +
five small new kernels (dynamic conv taps, the biased / sinked / windowed attention, the sigmoid router, the attention residual with the
loop gate, the SITU activation) - each checked against `ops_cpu` and the oracle before it is wired in.

## Code used from the DeepSeek branch (`upstream-dsv41`, `src/dsv41/`; allowed 22 Nov, "as required", the whole branch is to be ported here later)

Read from the branch without switching it. What matches Inkling, and what is done:

| DeepSeek code | Use for Inkling | State |
| --- | --- | --- |
| `route_kernel` (`ds4_kernels.cu`): biased top-k (selection key = score + bias, lowest index on a tie), weights = picked score / sum x route scale, a guard for non-finite gate input | Inkling's router has the same structure; the score is sigmoid, not sqrt-softplus | **hand-ported** as `src/kernels/cuda/inkling2_router.cu` (own small library `inkling2_kernels`, so Qwen's binary is unchanged), sigmoid scoring, a row stride of n (the original's is a fixed 1,024), without its `DS4_MASK` / vector-length extras; test `inkling2_router_parity`: on 6 layers x 48 tokens the picks equal the reference's seam (0 of 384 differ per layer), weights within 5.8e-7. **Found by that test, to check in DeepSeek's original at the full port:** its non-finite guard marks a pick with `-FLT_MAX`, which ties with the scan's start value, so with non-finite scores the same expert comes out repeatedly (indexes stay valid, as its comment promises, but they are not distinct); here a `taken` flag fixes it |
| `swiglu_kernel` and the clamped swiglu fused into `gate_up_kernel` | Inkling's SITU activation is the same place with a different body (tanh softcaps instead of a clamp) | the SITU is on the host in the NVFP4 check; the device version replaces the clamp's body |
| `gate_up_kernel<T, NE, RW>` / `down_kernel`: expert kernels templated on the weight type with `n_embd` / `n_ff` as run-time arguments | The pattern for a fused expert kernel that is not compiled for Qwen's 2560 / 640; DeepSeek's types are i-quants, NVFP4 would be a new `T` | not used yet: the base's `iq_mmvq(40)` + a small activation kernel is the verified path; fuse later if a profile asks |
| `route_ahead_kernel`, `route_publish_kernel` (look-ahead routing) | The loop-aware prefetch | to read when that step is reached |
| `attn_sink` (a learned logit per head), `sliding_window` | Inkling's sink is a projection of each token, not a constant | different: not reused |
| Its tiers (`ds4_store.hpp`, `ds4_pool.hpp`) | Not used: Inkling uses the base `core/` tiers | - |

**Debt to settle at the full DeepSeek port:** `inkling2_router.cu` and DeepSeek's `route_kernel` are then two routers that differ by one line. They become
one kernel with a scoring mode (qwertie's rule, which we share: do not copy a file to make a variant), and the `inkling2_router_parity` test stays as
its guard.

**How qwertie (Strataq) organized several models in one tree** (read from his `docs/ARCHITECTURE.md` on 22 Nov; nothing of his is copied, and we do not switch to his layout,
per the 21 Nov decision): layers, each using only the ones below. `src/base` (device runtime interface, platform services), `src/common` (model-independent:
artifact / GGUF, the expert store with RAM tier, VRAM slots, prefetch and eviction, kernels for any model, sampler, the serve loop, speculative decoding, conversation
state), `src/models/common` (geometry, layout tables, GDN / gated attention / MoE-routing layers shared by several families), `src/models/<family>` (`deepseekv41`,
`qwen36`, `qwen38`; a family never uses another), `src/program` (one thin `main()` each). Each engine program is a command loop plus a model behind a `PagedModel`
interface. His rules: do not copy a file to make a variant (share it, put the difference behind an interface); code in `common` / `models` does not call the CUDA
runtime directly; settings in a struct, not file-wide globals; `main()` parses arguments and calls a library. Ours already follows the folder-per-family shape
(`src/qwen35`, and `src/dsv41` on the DeepSeek branch), so Inkling gets `src/inkling2` and the shared parts stay in `core/` and the kernels; his paged engine and his
interface are not adopted (21 Nov: shared interface as new files, step 4, later).

## How Inkling gets ALL our features (found 22 Nov, read from both trees; the engine path starts here)

- **The Qwen3.8 tiers are wired to Qwen's engine, not to an interface.** `ExpertSource` is a clean interface, but `ExpertDispatch` (the per-layer
  VRAM hits, the CPU pool, the look-ahead tables `[n_layers][2][8][10]`) is driven from `generate.cpp` / `core/verify.cpp` / `layer.cpp` with Qwen's compiled
  geometry. Qwen3.6's path (`src/qwen35`, 4,813 lines) uses none of it (no `ExpertSource`, cache, stand-ins or KV streaming): it is **not** a template for features.
- **DeepSeek's `src/dsv41` is.** It is a standalone family path that already carries our features re-implemented and parameterised on the expert count:
  `ds4_store.hpp` (780 lines: the RAM tier as a slot arena with a replacement policy, whole experts by unbuffered reads, filled on demand and ahead of time by the
  look-ahead, blobs in the engine's gate | up | down layout, zero-copy to the GPU over PCIe when the arena is registered), `ds4_os.hpp` and `ds4_pool.hpp`
  (OS services and CPU workers; both carry Project Maya's MIT notice, so `third_party/project-maya/NOTICE.md` comes with them), `ds4_timeline.hpp`,
  `ds4_standin_tune.hpp` (572: the stand-in budget tuner, "the Qwen engine's rules ported to one token per step"), `ds4_lalearn.cu` / `.cuh` (the learned
  look-ahead ported from `native_router.cu`, rows padded to 512 with the real expert count passed in), `ds4_launch.h`, `ds4_experts_rule.hpp`.
  The closure is about 3,500 lines, mostly headers, with no dependence on DeepSeek's forward pass. **What is not in those files:** the VRAM slot tier, the ring and
  the lend live inside the 14.9K-line `ds4_model.cu` (store set-up at ~7532-7611, the per-token budget at ~11925): to be extracted, not copied.
- **Stages** (each ends with the feature gate: start-up feature lines equal to the clean build's, tokens identical, speed within noise, the area's tests pass):
  - **F1** bring the closure over to `src/dsv41/` under the same names (so the full DeepSeek port later is only adding files), with the Maya notice and their own tests.
  - **F2** the Inkling engine skeleton on the GPU: the reused kernels (conv, Q8_0, NVFP4 experts) + the new ones (dynamic taps, biased / sinked / windowed attention, attention residual and
    loop gate, SITU, L2 clip), experts read through the `ExpertStore` arena zero-copy (all in RAM): first end-to-end run, GPU log-probs against the oracle.
  - **F3** the VRAM slot cache and swaps, extracted from `ds4_model.cu` and keyed by (stored layer, expert).
  - **F4** stand-ins with the budget tuner; **F5** the learned look-ahead with the loop feature (the token's previous pass); **F6** KV: the ring for the 40 sliding visits, q8 KV
    for the 8 full ones; **F7** serve on 8095, speed against Maya's 32.4 tok/s, the batching test; **F8** Inkling-2-Small from the NVMe tier.

## F2 result (22 Nov): Inkling-2-Mini runs on our engine's GPU path

`src/inkling2/engine.cpp` (+ `include/strata/kernels/inkling2_kernels.hpp`, `src/kernels/cuda/inkling2_kernels.cu`; own libraries `inkling2_kernels` / `inkling2_engine`, so the
Qwen binary is unchanged) runs all 48 visits of a token on the GPU: Q8_0 dense weights and the Q8_0 / NVFP4 experts through the engine's `iq_mmvq`, the static short convs through the
engine's `gdn_conv_step`, RMSNorm through `rms_norm_weighted`, the router from the DeepSeek branch, and the new kernels (F32 matvec for the small tensors, Q/K/V preparation,
dynamic-tap conv, the biased / sinked / windowed attention, attention residual, loop gate, SITU). Experts sit in pinned, mapped host memory read from the GGUF (15.80 GB; 1.82 GB of dense
weights in VRAM; load 17-30 s) and the kernels read them zero-copy over PCIe: no VRAM cache yet (F3). That one access sits behind a single seam for F3 to replace with the `ExpertStore`.

Against the float64 reference on the oracle prompts (teacher-forced, `gpu_forward_test`):

| prompt | top-1 same | mean KL (reference \|\| engine) | worst visit `out` | picks differing | speed |
| --- | --- | --- | --- | --- | --- |
| 48 tokens | 45 of 48 | 0.0050 | 7.9% of its max | 1,488 of 18,432 | 9.3 tok/s |
| 92 tokens, window 32 / extent 48 / floor 16 | 88 of 92 | 0.0059 | 7.1% | 2,783 of 35,328 | 9.3 tok/s |

Where the drift comes from (read from the seams, not assumed): at the first visit every attention-path seam (`q`, `k`, `v`, `attn_gated`, `attn_out`) is within 0.2-0.5% of the reference, which is
the floor of Q8_0 weights times int8 (q8_1) activations; the MoE output then shows 3.7-4.7% because ~3.6% of the router's picks swap where two experts are near-tied (the picks are compared
exactly in the CPU oracle, which matches). So the GPU path's difference is int8 activation noise amplified by routing, the same envelope every expert on this engine has, not a kernel fault.
The speed is the naive path's (no cache, 8 experts x 3 launches a layer visit, host sync for the picks): the starting point for F3 and the grouped kernels.

## Maya's decode path traced (`maya-ik2`, `src/ik2/ik2_main.cu`, `forward`) against ours (22 Nov)

Why Maya gives ~32 tok/s on this 12 GB card where F2 gave 9.3: Maya's author says a token costs its weight bytes plus its launches. Our F2 profile (`STRATA_INKLING2_PROFILE=1`, per token, a sync between
phases) says where ours goes:

| ms a token | attention side | router + host round trip | experts | tail | head | total |
| --- | --- | --- | --- | --- | --- | --- |
| F2, no cache | 14.6 | 1.4 | **88.0** | 1.0 | 1.6 | 106.5 (9.3 tok/s) |
| F2, experts of layers 0-13 in VRAM (`STRATA_INKLING2_VRAM_LAYERS=14`, identical numerics) | 14.5 | 1.3 | **43.8** | 0.9 | 1.6 | 62.1 (15.7 tok/s) |

The experts are 82% of a token: 1.85 GB of expert bytes a token (44 NVFP4 visits x ~36 MB + 4 Q8_0 visits x ~68 MB) read zero-copy at ~21 GB/s, the PCIe 4.0 x16 link's practical rate. With 14 layers resident the
traffic falls to 0.77 GB but the experts phase only halves: the resident experts' kernels and the zero-copy ones run one after another on one stream, 8 experts x 6 launches a visit (~2,300 launches a token).

| | Maya's path (read in the code) | ours now | to match (reuse first) |
| --- | --- | --- | --- |
| Embedding | the row decoded **on the host** from the mapped GGUF and copied: no 0.5 GB table in VRAM | Q8_0 table in VRAM (0.49 GB; the head's 0.49 GB too) | host embedding row: ~0.5 GB more for the expert cache (small, do first) |
| Dense math | weights decoded to FP16 or planar Q8, FP16 activations; fused launches (`mv_q8p_conv`: the conv rides the product; q / k / v / gate one launch `MvAlt`; the three norms one; the sink rides the clips) | iq_mmvq with int8 activations, ~40 launches a visit | fuse; a CUDA graph for the stretch between routing syncs. Accuracy is already on par: Maya's own engine vs the float64 reference on Mini is 45-46 of 47 top-1, KL 6-7e-3; ours 45 of 48, 5.0e-3 |
| Router | `gemm16` + `route`, the picks copied async to the host, **one stream sync a visit** (the routing round trip) | the same, two small syncs a visit (1.4 ms a token) | one sync |
| Expert cache | a VRAM slot cache (`ExpertCache`, a tick for its policy); each pick looked up on the host: **hits** use their slot, **misses** copy into a slot on a **second stream** (`g_cs`) while the hits' kernels run (events order them) | none: every expert read over PCIe | **F3**: the VRAM slot tier extracted from the DeepSeek path (`ds4_model.cu`) behind the engine's one seam, with the copy stream and the overlap; our own look-ahead |
| Misses | the CPU lane (`IK2_CPU`, NVFP4 AVX-512) computes a missing expert when sooner than its copy (0.5 ms against 0.36 ms) | none | the engine's own NVFP4 CPU kernels (`nvfp4_avx2` here: the i5-12400F has no AVX-512), as a later arm |
| Expert kernels | **grouped**: `experts_gate_up_swiglu` (gate + up + the activation, all of a group's experts, one launch from a pointer array) and `experts_down` (down + combine) | 6 launches an expert | grouped kernels in the DeepSeek branch's `gate_up_kernel<T, NE, RW>` pattern (runtime shapes, the activation fused) with NVFP4 / Q8_0 as the type and SITU as the body; checked against the oracle |
| Prefetch | the next layer's likely experts while this one runs; two or three layers ahead, or predicting from this layer's input (46% precise), **did not help** | none | our learned look-ahead, plus the loop's own signal (the token's picks of the pass before: 4.9 of 8 shared) |
| Resident path (everything in VRAM, a 24 GB card) | device-side routing (`route_token`), a CUDA graph a token, 757 launches, 124.8 tok/s on a 3090 | n/a on 12 GB | later, for cards that fit it |

**Projection from the profile** (a plan, not a result): experts 88 -> ~20 ms needs about 85% of the expert bytes to hit the cache plus overlap (a ~7 GB cache holds 44% of the experts by bytes; the loop gives the cache a
62% head start on passes 2 and 3), the attention side 14.6 -> ~7 ms from fusion; with ~4 ms of round trips, tail and head that is ~31 ms a token = Maya's ~32 tok/s, and our own features
(look-ahead, the stand-ins) are what can go beyond it. Order: grouped expert kernels, then F3's cache with the copy stream, then the fusion; each measured against `gpu_forward_test` and the oracle.

## Purpose and method: the speed ledger (stated by the owner, 22 Nov)

The point of porting Inkling is **not the port**. It is to find the speed our engine loses on capable systems, where the model fits RAM + VRAM and the SSD should never be touched (the NoPlaceLikeLocalhost video:
HotRod 18 -> 36 tok/s on a starved 12 GB card, but 103 -> 87 on a roomy 24 GB one). Inkling-Mini is the lab: class B on the 3060 box, and class A on a 24 GB card, with an outside engine (Maya's `maya-ik2`: 32 tok/s here, 112-125 on a
3090) as the ceiling to measure ourselves against. So every stage is a measurement, recorded here as a ledger, and each finding is mapped back to the Qwen, Edge0 and DeepSeek engines.

**The ladder**, one step at a time, each measured on the same prompts and checked against the oracle (tok/s and KL):

1. **L0** the F2 engine: 9.3 tok/s (experts 88 ms of 106.5: all 1.85 GB a token over PCIe).
2. **L1** + grouped expert launches (one launch for a group's gate + up + SITU, one for down + combine).
3. **L2** + the VRAM slot cache over the RAM arena, misses copied on a second stream while the hits compute. This is the class-B baseline and should land near Maya's 32 tok/s.
4. **L3** + launch fusion and a CUDA graph for the dense stretch between routing syncs.
5. Then **our features, added ONE at a time on top of L3**, each with its tok/s delta and KL delta: pinned segments, the engine's own look-ahead, the loop-aware prefetch, the exchange / adaptive swaps, the tuner,
   the stand-ins, KV ring / q8. **A feature that costs speed when the model fits is recorded as such** (the verdict column) - that is the loss the video showed, found where the SSD cannot hide it.
6. Same ladder on a 24 GB card (class A: everything resident, Maya's graph path as the ceiling) when one is available; the 2x / 4x batching-scale test on class B.

| step | tok/s | KL vs oracle | what it showed |
| --- | --- | --- | --- |
| L0 F2 | 9.3 | 5.0e-3 | experts 82% of a token, PCIe-bound; attention side 14.6 ms, router round trip 1.4 |
| L0 + 14 layers' experts in VRAM | 15.7 | 5.0e-3 | experts 88 -> 43.8 ms, not 88 -> 37: the resident and zero-copy kernels serialize on one stream, ~2,300 launches a token |
| Maya (reference) | ~32 | 6-7e-3 (its own, Mini) | cache + copy stream + grouped kernels + fused launches; no stand-ins, no SSD |

## Rules for this port

- The Qwen path, its layer-major prompt walk and its prefill speeds are not touched. Measured before any shared file changes:
  13.5K-token prompt read 336 / 333 tok/s, decode ~20 tok/s (22 Nov, clean main).
- No copied engine code. Reading Maya's `src/ik2` to understand a seam is fine; credit where an idea comes from it.
- Hand-port in small commits; every commit builds; parity numbers go in the commit message.

## Credits

Maya (mw00 / peasantsmith): the `inkling2` specification (`docs/INKLING2.md`), `ref/inkling2.py`, and `maya-ik2`, with its
contributors @alesha-pro, @mrsirg97-rgb, @OscarSheng and @cobra91 as named in Maya's v1.1.72 changelog. The model is
thinkingmachines' Inkling-2-Mini (Apache 2.0); the GGUF layout is llama.cpp's `inkling2` conversion as shipped by unsloth.

## L1 result: grouped experts as a shared library (22 Nov)

`include/strata/kernels/moe_group.hpp` + `src/kernels/cuda/moe_group.cu` (library `strata_moe_group`) hold the router (score mode: sigmoid or DeepSeek's
sqrt-softplus) and the grouped experts (`experts_prepare / gate_up / down / combine`): run-time shapes, NVFP4 and Q8_0 decode, the activation a parameter
(SwiGLU, DeepSeek's clamped SwiGLU, Inkling's SITU), optional per-expert scales, experts reached by a pointer array (any tier that can give an expert's address can
drive them). Nothing Inkling-specific is in them, so Qwen / Edge0 / DeepSeek can call them behind their own switches; none does yet (the feature gate comes with that).
`moe_group_parity` checks them against the CPU oracle: layer 0 NVFP4 and layer 20 Q8_0 rel ~2e-7, all three activations, both router scores.

Engine change: one `route` + 4 launches a visit instead of 6 launches per expert; picks stay on the device and the host reads them once a token.

| step | all experts over PCIe | experts of layers 0-13 in VRAM |
|---|---|---|
| L0 (per-expert launches, int8 activations) | 9.3 tok/s | 11.4 |
| L1 (grouped, FP32 activations) | 9.8 tok/s | 16.0 |

- All-over-PCIe barely moves: it is bandwidth-bound (1.85 GB/token at ~21 GB/s), as the profile said. The launch savings show once bytes are not the limit (+40%).
- Lesson: the first dot mapping (byte loads per lane) lost 40% over PCIe (117 ms experts); 32-bit/16-bit per-lane loads restore it. Wide loads matter for zero-copy.
- KL against the oracle varies 0.0034-0.0073 with the summation order of the dots: near-tie router picks (~6% of picks differ), the same envelope as Maya's engine, not an error.
- Next: L2 (VRAM slot cache + copy stream, class B), then L3 (fusion / graph).

## L2 result: VRAM slot cache, class B (22 Nov)

`include/strata/kernels/moe_slot_cache.hpp` (`moe::SlotCache`, header-only, no model's numbers): the caller gives layers, experts, the bytes of an expert's two planes and where an
expert's bytes are on the host; per visit `place()` copies the misses into least-recently-used VRAM slots on a copy stream (a hit costs nothing) and returns slot addresses for
`moe::experts_gate_up / down`. Switches are model-neutral: `STRATA_MOE_SLOTS=N` (0 = off, unset = what free VRAM holds less `STRATA_MOE_SLOT_MARGIN_MB`, 1500).
The engine reads the router's picks once a visit (one sync), then the kernels read slot addresses. Not tied to Inkling: any engine whose experts sit in pinned RAM can use it.

| step (48-token oracle prompt, cold cache) | tok/s | experts ms/token |
|---|---|---|
| L1, all experts zero-copy over PCIe | 9.7 | 82 |
| L2, slot cache 1036 experts (8.8 GB, 34%), 86% hits | 22.3 | 24 |

KL against the oracle is identical with and without the cache (0.0046): the cache moves bytes, not values. Run is cold (48 tokens), so steady-state hit rate is higher.
Not yet done: overlap of the copies with compute (needs the picks before the visit: look-ahead / loop-aware prefetch, our idea to add one at a time), L3 fusion / graph.

Qwen check after the change (3 prompts, Strata-edge0 binary, 262144): 18.0 / 18.9 / 21.0 tok/s (mean 19.3, clean main 19.3), 13.6K-token read 339.7 tok/s (clean main 338.1):
unchanged, as expected: neither library is linked into the `strata` binary. Qwen's own types (IQ) are not decoded by `moe_group` yet.

## L3a result: crude look-ahead prefetch as the control arm (22 Nov)

`STRATA_MOE_PREFETCH=N` (default 0, off): at a visit's router the NEXT visit's router is run on this visit's input (its own norm) and the N best experts not in a slot are copied
on the copy stream while this visit computes (`SlotCache::prefetch`; never evicts a slot of the current visit). This is the "predict from this layer's input" idea, not our
learned look-ahead: it is the control the learned look-ahead (`ds4_lalearn`, 89 / 80 / 72% at +1 / +2 / +4 layers on Qwen) and the loop feature must beat.

| arm (48 tokens, cold, 1036 slots) | picks named | hit rate | PCIe copies (misses + prefetch) | tok/s | ms/token |
|---|---|---|---|---|---|
| no prefetch | - | 86.2% | 2548 | 22.3 | 43.0 |
| N = 8 | 75.5% | 94.5% | 1015 + 2557 = 3572 | 21.9 | 43.7 |
| N = 12 | 87.2% | 96.0% | 731 + 4369 = 5100 | 20.2 | 47.7 |

KL identical (0.0046): placement only. Findings:
- On a looped model with an 8-byte-wide window the crude predictor is far better than the 46% Maya measured (it also covers passes 2 and 3, which the cache already holds): 75-87%.
- It still does not help: the hit rate rises, but the PCIe bytes do not fall (wasted prefetches add copies) and the window is short (the next visit's picks are known only one visit
  ahead; the copy of a few experts takes about as long as the compute it overlaps). Same verdict as Maya's.
- What would help: a longer reach (+2 / +4 visits, which is what the learned look-ahead is for), fewer wasted copies (confidence, not top-N), and for passes 2-3 protection of the
  token's pass-1 experts rather than prediction (about 96 experts, 430 MB, fit the cache).

Qwen check (3 prompts): 17.3 / 19.8 / 22.6 tok/s (mean 19.9, clean main 19.3), 13.6K-token read 343.1 tok/s (338.1): unchanged.

## Why prefetch cannot pay yet: the floor (22 Nov)

Two checks on the control arm above. (1) Without the profiler (its `lap()` syncs the whole device, which also waits for the copy stream) the arms read the same: 22.1-22.3 tok/s without
prefetch, 22.3 (N=4), 22.5-22.9 (N=8), 20.2-20.8 (N=12). (2) `STRATA_INKLING2_BENCH=1` in `gpu_forward_test` times the engine alone (no seam copies - each "out" seam was a synchronous
copy per visit - and no host softmax), cold cache then warm:

| 48 tokens | ms/token | tok/s |
|---|---|---|
| cold cache | 42.7 | 23.4 |
| warm cache (same tokens again: the floor, few misses) | 39.7 | 25.2 |
| every expert zero-copy over PCIe | 99.5 | 10.0 |
| cold, prefetch N=8 | 41.9 | 23.9 |

The slot cache already removed the PCIe wait: misses cost ~3 ms of 42.7. The token is now **compute / launch / sync bound at ~39 ms** (attention side ~15, experts ~21 = 0.44 ms a visit
where the bytes (~36 MB from VRAM at ~360 GB/s) need ~0.1, the rest launches, the per-visit sync and kernel latency, plus head, tail). A perfect prefetch can save at most ~3 ms (7%).
Prefetch pays when the engine is transfer-bound: (a) after the compute side is cut to ~20 ms (then the ~16 ms of PCIe a token at a 34% cache is no longer hidden by it), (b) on a
smaller card or a bigger model (Inkling-2-Small, class C), where NVMe / PCIe is the wall - that is where our learned look-ahead and the loop feature are for. Order: cut the 39 ms first.

## L3b: where the 39 ms goes (nsys, 22 Nov) and the stream fix

Per token (144 tokens profiled, 2,030 launches a token): GPU kernel time ~27 ms of 40-43 wall. `k_gate_up` 7.9 ms (165 us x 48, ~145 GB/s of ~360), `k_down` 4.7, dense Q8_0
`mmvq` 6.1 (289 launches; the 200K-row head is 1.4 ms of it), `k_matmul` (router, dyn-conv projections) 3.4, `attn_res` 1.6, the norms 0.85, the rest small.
**Found: ~371 `cudaDeviceSynchronize` a token.** The legacy kernel wrappers (`rms_norm_weighted`, `gdn_conv_step`, the elementwise ops: `sync_if_needed`, `gdn.cu`) synchronize the whole device
after every launch when given `stream == nullptr`, and the engine passed nullptr everywhere. The engine now runs on one explicit stream (any engine that passes nullptr to these wrappers
pays the same). Cold 48 tokens, engine alone: 42.7 -> 41.1 ms (23.4 -> 24.3 tok/s); with look-ahead prefetch N=8: 38.5 ms (26.0 tok/s, +6.5%): prefetch starts to pay as the compute shrinks.
KL unchanged (0.0046). Next: the expert kernels' memory-level parallelism (12.6 ms at ~40% of VRAM bandwidth), the per-visit host gap, fusion of the small launches.

Unrolling the expert dot loops (`#pragma unroll 4`, several blocks' loads in flight): cold 41.1 -> 40.4 ms (24.8 tok/s), with prefetch N=8 38.0 ms (26.3 tok/s); `moe_group_parity` unchanged
(2e-7). Small: the expert kernels are not limited by loads in flight alone; the remaining ~13 ms between kernel time (~27 ms) and wall is the per-visit host gap (router picks to host,
slot placement, then the expert launches) and the ~2,000 small launches a token. Qwen check after this change (3 prompts): 19.8 / 18.7 / 20.6 tok/s (mean 19.7, clean main 19.3), 13.6K read 333.5 tok/s (338.1; run-to-run spread 333-343): unchanged.

Spin instead of sleep while the host waits for the router's picks (`cudaDeviceScheduleSpin`, `STRATA_INKLING2_BLOCKING_SYNC=1` turns it off): cold, no prefetch 40.9 -> 40.1 ms; with
prefetch N=8 38.0 -> 37.9 (noise-sized, kept on). The per-visit host gap is not wake-up latency. What is left between ~27 ms of kernel time and ~38-40 ms of wall is the host-side work
between the picks arriving and the expert launches going out, and ~2,000 launches a token; the weight bytes alone (~2.3 GB a token) would need ~8 ms at VRAM speed, so the kernels are
also well off the bandwidth floor (experts 12.6 ms for 1.85 GB). Qwen check (3 prompts): 17.1 / 18.9 / 23.0 tok/s (mean 19.7), 13.6K read 342.5 tok/s: unchanged.

## L3c: the expert kernels were ALU-bound; int8-activation path (22 Nov)

The exact FP32 expert kernels spend ~12 integer ops per nibble in `e2m1d`; the 165 us a `k_gate_up` was decode, not bytes. `moe::quantize_act` + `moe::experts_gate_up_q8 / experts_down_q8` quantize the
activation to q8_1 blocks and decode the weights with the byte-permute table lookup + `__dp4a` (the engine's `iq_kernels.cu` lookup, ggml MIT; same NVFP4 word layout; Q8_0 by dp4a directly).
`STRATA_MOE_INT8` (engine default ON, `=0` the exact FP32 path; the library keeps both). Kernel parity vs the CPU oracle: 8.5e-3 (layer 0 NVFP4) and 1.3e-2 (layer 20 Q8_0) relative, bound 3e-2;
the FP32 path stays 2e-7.

| 48 tokens, cold, engine alone | ms/token | tok/s | KL vs oracle |
|---|---|---|---|
| FP32 experts | 40.7 | 24.6 | 0.0046 |
| int8 experts | 35.8 | 27.9 | 0.0051 |
| int8 + look-ahead prefetch 8 | 33.5 | 29.8 | 0.0051 |
| same, warm cache (floor) | 32.0 | 31.2 | 0.0051 |

Top-1 identical (44 of 48). Maya's engine: ~32 tok/s on this card. The KV cache and conv state stay FP32 (only the expert FFN input and activation are int8; the dense q/k/v/out
projections already used int8 activations in the first GPU engine). Qwen check (3 prompts): 18.7 / 18.6 / 22.8 tok/s (mean 20.0), 13.6K read 342.0 tok/s: unchanged.

## L4: the VRAM sizing follows the Qwen engine's rule; embedding on the host; first serve (22 Nov)

- **VRAM cache sizing (not a guess any more):** slots = (free VRAM - reserve) / slot bytes, read after every other engine allocation and after one warm-up token (the runtime allocates lazily), reserve =
  `STRATA_VRAM_RESERVE_MIB` (default 700, the Qwen engine's `--vram-reserve-mib` default). Qwen additionally adds the prompt path's buffers, the draft head and the pipeline windows to the reserve and learns a per-machine
  reserve (`vram-reserve.json`); those two terms are not in the Inkling engine yet (no prompt path / draft head here).
- **Embedding table on the host** (pinned, zero-copy row read; `STRATA_INKLING2_EMBD_VRAM=1` keeps it in VRAM): +0.49 GB for the cache. Cache 846 -> 1193 experts (39%) at the serve context of 4096; cold 31.3 tok/s, warm 33.1 tok/s with
  prefetch 8 (engine alone, 48 tokens; Maya ~32), 29.7 / 32.2 without prefetch. KL 0.0051.
- **Served end to end** through the Python server's protocol: `inkling2_serve --serve` (INFO / READY, GEN, PP, T, DONE, STOP, QUIT; conversation continuation reuses positions). Greedy answer "The capital of France is Paris."
  (reasoning channel and answer both right); ~28 tok/s generation through the server, prompt read ~27 tok/s (one token at a time: no batched prefill yet). Maya's v1.1.72 server carries the Inkling reply parser
  (`serve/frontend.py`: content_thinking / content_text, stop on `<|content_model_end_sampling|>`); ours does not yet, so for now it is the front of our engine on 8095 (port the parser into our `serve/`).
- **Known departures from our design, to be replaced (owner, 22 Nov: "use our design")**: (1) the host tier is a pinned `cudaHostAlloc` + `pread` copy, not `ds4::ExpertStore` (src/dsv41/ds4_store.hpp: unbuffered reads that
  leave the page cache alone, a policy-owned arena, `LayerGeo` per-matrix offsets and strides, pinned chunks, guesses read ahead); (2) the VRAM tier is my `moe::SlotCache` (LRU), not the engine's `core::ExpertCache` / the DeepSeek
  branch's VRAM tier, and its slots are sized for the Q8_0 layers (8.4 MB) although 20 of 24 layers are NVFP4 (4.5 MB): about half the cache is wasted; (3) no stand-ins, no learned look-ahead, no adaptive swaps yet.
  Plan: two stores (NVFP4 layers, Q8_0 layers: `ExpertStore` slots are uniform), `LayerGeo` for the fused gate_up (gate and up are two offsets of one tensor, stride = the expert plane) and the down tensor, sized VRAM slots, then the
  look-ahead and the loop feature on top.
Qwen check after this change (3 prompts): 16.7 / 20.6 / 21.5 tok/s (mean 19.6), 13.6K read 334.7 tok/s: unchanged (clean main 19.3 / 338.1; run-to-run spread 333-343).

## Served by HotRod's own server (22 Nov)

`serve/inkling.py` (new): `InklingParser` rewrites Inkling's typed segments (`<|message_model|><|content_thinking|>` / `<|content_text|>` / `<|content_invoke_tool_json|>` ... `<|end_message|>`,
turn end `<|content_model_end_sampling|>`) into what `OutputParser` reads (`</think>`, the `<tool_call><function=..><parameter=..>` form), so streaming, tool-call events, stop strings
and usage stay the server's own; `ChatTemplate` detects Inkling's template and converts messages (tool-call arguments as objects, developer -> system) and the thinking kwargs
(enable_thinking=False -> effort "none"); `end_token_ids` also stops on `<|content_model_end_sampling|>`. The segment grammar and the rewrite-before-the-parser idea are Project Maya's
(`serve/frontend.py` `InklingRewriter`, MIT, credited in the module and third_party/project-maya/NOTICE.md); the code is written for our parser (no THINK_AGAIN: a reasoning segment after
the answer began reads as answer text). `serve/test_inkling.py`: reasoning / answer split at every chunk size down to one character, a plain answer, a tool call, messages and kwargs.
Live on 8095 through `python -m serve.server --engine strata` with `inkling2_serve` as the engine: `content` "The capital of France is Paris.", `reasoning_content` the thinking, finish `stop`;
200-token answers in 7.6-8.0 s including a ~30-token prompt read (~30 tok/s generation). Context 4096 (the KV is FP32 for every visit; the sliding-window ring is next).

## Qwen decode profile and what the Inkling steps map to (22 Nov, nsys with CUDA-graph nodes traced: `--cuda-graph-trace=node`; without it the graph's kernels are invisible)

Qwen3.8 UD-Q4_K_XL, default config, 3 x 300-token answers (16-19 tok/s under nsys). Per steady decode window (300 ms): the main stream is busy 27%; three fetch streams
(`fetch_blobs_kernel`, then `native_gu_kernel` / `native_down_kernel` on the fetched experts, with `wait_flag_ge_stamped_kernel` spins between: verify.cpp, the PCIe-balance flags) 13-17% each;
some kernel runs 72% of the window, and without the wait kernels 46%. Kernel time over the whole decode: wait_flag 43%, fetch_blobs 15%, the compute kernels ~10%.
Nothing of the Inkling compute work applies to those numbers: its grouped `native_gu/down/mmvq_multi` kernels are already dp4a / q8_1 (L1, the int8 path), the cache and the in-graph device-side fetch
exist (L2), the learned look-ahead exists (prefetch), decode is in CUDA graphs (no per-launch syncs), the embedding row is already host-side (`native_head.hpp`), the cache is sized free - reserve.
The headroom is the half of each token the GPU spends waiting on the miss path (CPU lane + fetch). Candidates from the Inkling twists, to be built as new code and measured one at a time on the
3-prompt check: copy-engine DMA for the predicted experts ahead of the layer (Inkling's copy stream) instead of an SM-based fetch in the graph; the loop-style protection of an expert across a token.
The learner (`la_b_*`, ~1% of the window on its own stream) is not on the critical path in the window measured.
