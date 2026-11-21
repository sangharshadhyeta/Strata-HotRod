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

## Rules for this port

- The Qwen path, its layer-major prompt walk and its prefill speeds are not touched. Measured before any shared file changes:
  13.5K-token prompt read 336 / 333 tok/s, decode ~20 tok/s (22 Nov, clean main).
- No copied engine code. Reading Maya's `src/ik2` to understand a seam is fine; credit where an idea comes from it.
- Hand-port in small commits; every commit builds; parity numbers go in the commit message.

## Credits

Maya (mw00 / peasantsmith): the `inkling2` specification (`docs/INKLING2.md`), `ref/inkling2.py`, and `maya-ik2`, with its
contributors @alesha-pro, @mrsirg97-rgb, @OscarSheng and @cobra91 as named in Maya's v1.1.72 changelog. The model is
thinkingmachines' Inkling-2-Mini (Apache 2.0); the GGUF layout is llama.cpp's `inkling2` conversion as shipped by unsloth.
