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

## Rules for this port

- The Qwen path, its layer-major prompt walk and its prefill speeds are not touched. Measured before any shared file changes:
  13.5K-token prompt read 336 / 333 tok/s, decode ~20 tok/s (22 Nov, clean main).
- No copied engine code. Reading Maya's `src/ik2` to understand a seam is fine; credit where an idea comes from it.
- Hand-port in small commits; every commit builds; parity numbers go in the commit message.

## Credits

Maya (mw00 / peasantsmith): the `inkling2` specification (`docs/INKLING2.md`), `ref/inkling2.py`, and `maya-ik2`, with its
contributors @alesha-pro, @mrsirg97-rgb, @OscarSheng and @cobra91 as named in Maya's v1.1.72 changelog. The model is
thinkingmachines' Inkling-2-Mini (Apache 2.0); the GGUF layout is llama.cpp's `inkling2` conversion as shipped by unsloth.
