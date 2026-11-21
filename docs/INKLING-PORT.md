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
- **Not covered by that oracle:** the full layers' log scaling past 8,192 positions (tau is 1.0 below that), sliding windows longer than
  48 tokens (the window is 1,024), and the relative-bias extent past 48 positions. A long-prompt oracle is needed before those are claimed.
- **Next:** (a) a long-context oracle; (b) incremental decode: KV for the 8 full-attention layers' visits and the sliding window, the conv
  states (3 rows per conv), per-position block states; (c) the CUDA kernels, each checked against `ops_cpu`; (d) the expert tiers.

## Rules for this port

- The Qwen path, its layer-major prompt walk and its prefill speeds are not touched. Measured before any shared file changes:
  13.5K-token prompt read 336 / 333 tok/s, decode ~20 tok/s (22 Nov, clean main).
- No copied engine code. Reading Maya's `src/ik2` to understand a seam is fine; credit where an idea comes from it.
- Hand-port in small commits; every commit builds; parity numbers go in the commit message.

## Credits

Maya (mw00 / peasantsmith): the `inkling2` specification (`docs/INKLING2.md`), `ref/inkling2.py`, and `maya-ik2`, with its
contributors @alesha-pro, @mrsirg97-rgb, @OscarSheng and @cobra91 as named in Maya's v1.1.72 changelog. The model is
thinkingmachines' Inkling-2-Mini (Apache 2.0); the GGUF layout is llama.cpp's `inkling2` conversion as shipped by unsloth.
