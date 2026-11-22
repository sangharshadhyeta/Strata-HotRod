# Learnings from other implementations that we can implement

1 Oct 2026. Only techniques we do not have, or have a worse version of, written so each can be built. Sources were read in full from their papers; every
number marked "source" is THEIR claim on THEIR hardware, not measured here. Each item names where it goes in our code, the test, and what decides whether it
becomes a default (paired against the shipped default, same-build control, fidelity harness for anything that changes outputs).

Not on this list because ours is already as good or better: atlas-similarity stand-ins (BuddyMoE pairs), lookahead + next-layer predictor (Fate/HOBBIT prefetch),
heat/LFRU eviction (LRU/LFU/ARC), lazy KV commit (vAttention; we only add the reclaim, see L7), whole-expert admit/evict (Strata; gemma measures 5.1% partial residency).

## L1. Bias the router toward resident experts (Cache-Conditional Experts, arXiv 2412.00099) - IMPLEMENTED, tested on both models, not yet defaulted

- Mechanism (source): `z' = z + lambda * delta_avg * m_t`. We implemented the simpler multiplicative form: resident-expert scores x(1+beta) before top-k
  (`LLAMA_MOE_CACHE_BIAS=beta`), forward-pass weights still computed from the UNBIASED scores, so every chosen expert is computed exactly. Lives in
  `src/llama-graph.cpp` (`llm_graph_input_moe_cache_bias`) + `ggml_backend_cuda_moe_cache_resident_counts` in moe-cache.cu (one pass over each pool's map
  under the session lock, rebuilt per decode). **Decode-only** (`LLAMA_MOE_CACHE_BIAS_MAX_BATCH`, default 8 tokens) - a first version biased prefill too by
  mistake, which has no cache-residency benefit and only changes the whole prompt's hidden state for nothing; fixed in `391f116f0`.
- RESULT (see ROADMAP.md "Router bias vs substitution" for the full table): bias measurably raises cache hit rate on BOTH gemma-4 (0.847->0.917 at beta=1.0)
  and Qwen3.8-Flash-Next (0.26->0.33 at beta=0.5), which substitution provably cannot do. Substitution's own documented +8.11% win was on a different model
  (Ornith-1.5-35B, 256 fine-grained experts) that was never re-tested here - so this isn't "bias beats substitution," it's "substitution was never actually
  validated in gemma's or Qwen's regime, and bias is the first thing that measurably helps the cache-miss problem in THAT regime."
- NOT a default yet: kept opt-in, substitution kept too (not deleted). Pending a clean Qwen speed/fidelity rerun (the one that ran had disk I/O contention
  from an unrelated bug) and an owner decision, since this changes default output quality on every request.

## L2. Importance by cumulative gate mass, with a skip class (HOBBIT, arXiv 2411.01433) - SCOPED, blocked on a plumbing change

- Mechanism (source): rank the K selected experts by gate weight; unimportance `s_i = sum of the weights of all higher-ranked experts`. `s <= 0.6` full precision,
  `0.6 < s <= 0.9` low precision, `s > 0.9` SKIPPED entirely. Gate magnitude correlates 0.99 with the expert's actual output contribution. Mixtral: 67% full / 30% low / 3% skipped; accuracy loss <1%.
- What we do instead: a rank floor (substitute only ranks >= 6) - rank ignores how much weight the expert carries. On gemma-4 ranks 4-7 hold only 32% of the gate mass.
- CONFIRMED BY READING THE CODE (2 Oct): `rank_bucket` in `moe_cache_plan_impl` is PURELY POSITIONAL (`index % rank_top_k`, i.e. "which slot in the
  top-k list"), computed from `ids[]` alone - no gate-weight array reaches the planner at all. This is a bigger change than the gemma-4 substitution code's
  existing rank-based gating: it needs a parallel float-weights array threaded through the op callback interface that feeds `moe_cache_plan_impl`,
  touching every call site in the hot MoE dispatch path. Deliberately not attempted at the end of a long session (see ROADMAP.md rules) - the risk of a
  half-verified change to a path every model goes through outweighs getting this one learning in before stopping.
- Where: the op callback signature feeding `moe_cache_plan_impl` (search call sites of that function) would need a `const float * weights` parallel to
  `ids[]`. Once that plumbing exists: the substitution gate in `moe_cache_plan` / `moe_cache_substitute_min_rank` -> a cumulative-mass threshold; a missed
  expert past the skip threshold is dropped (weight 0, no fetch, no CPU compute) instead of substituted.
- Test (once built): replace the rank floor by `s` thresholds (0.6/0.9 first, then sweep); fidelity + paired speed on the test worktree BEFORE touching main.
- Second half (later): a low-precision copy of an expert held in RAM, loaded instead of the full one for the middle class. Needs a requantized copy (extra RAM); only after the first half is measured.

## L3. Adaptive draft length from a cost model (MoE-SpeQ, arXiv 2511.14102)

- Mechanism (source): choose k to maximise `E[accepted tokens] / E[cycle time]`, `cycle(k) = max(T_draft(k), T_io_init) + T_io_new(k) + T_verify(k+1)`, where
  `T_io_new` grows with the number of NEW experts the k drafted tokens need. Acceptance tracked by EMA, costs profiled offline and updated online.
- Why it matters here: depth 8 loses (-11%) and depth 4 wins because verify width widens the expert union; the best k differs by prompt (accept 0.87 code, 0.67 prose). We fix 4.
- Where: `common/speculative.cpp` / the draft loop in `common/sampling.cpp` (n_max, p_min). Track acceptance per draft position (EMA) and the measured verify time per width; pick k each round.
- Test: mixed and calibration sets paired against fixed 4; must not lose on code and should gain on prose. Cheap, gemma-testable, no new kernels.

## L4. Fate's cache and prefetch details (arXiv 2502.12224)

- Shallow-layer-favoring allocation (source): layers 0-3 predict poorly (~60-70% prefetch accuracy vs ~85%+ deep), so cache ALL experts of the first L layers and split the rest
  evenly: `cache_deep = (cache_total - L*n_expert) / (n_layer - L)`. Source hit rate 99.08%. Ours: one shared heat ordering. Test: heat bonus (or reserved slots) for the first N layers; per-layer hit-rate lines already exist in the SUMMARY output.
- Confidence filter on prefetch (source): transfer only predicted experts above the 75th percentile of confidence: accuracy 78.8% -> 97.2%. Ours admits predicted experts into free slots without a confidence cut. Test: add the percentile filter to lookahead/predictor admission; measure wasted fills and hit rate.
- Time-budgeted prefetch (source): `n = T / t_expert`, `T = t_moe + t_attn + t_gate` per layer: only as many experts as can transfer while the layer computes. Ours caps by ring size and free slots, not by time. Test: cap fills per layer by measured layer time over measured fill time.
- Prefill order (source): transfer experts by descending token count (most-demanded first). Where: `moe_cache_prefill_advance` order. Test: first-token latency on 5K-token prompts, which is what ThinkVest sends.

## L5. SSD-LLaMA's read pipeline (arXiv 2609.18110) - for Qwen, where misses come from the SSD

- Mechanism (source): many independent reads in flight in bounded batches with O_DIRECT, each completed read goes to VRAM asynchronously without waiting for the rest; one aligned block per expert;
  all prompt rows for an expert form one task so each expert loads once per prefill pass; RAM-resident experts run on CPU or on GPU after H2D by measured end-to-end latency. Source: SSD bandwidth 77.7% of peak vs 43.2% for llama.cpp; decode 2.1-15.6x (PCIe 5 SSD, RTX 5090).
- What we have: one `pread` per expert behind `GGML_CUDA_MOE_CACHE_EXPLICIT_READ=1`, called from a single fill worker; misses are otherwise read by CPU threads page-faulting the mmap.
- Plan: (1) MEASURE first - NVMe read MB/s and queue depth during Qwen decode (iostat) against the drive's rated speed, and `PART_STATS` miss MiB/token; if the SSD is not near its limit this item is moot.
  (2) At plan time, with the router result known, issue ALL missed experts of the layer at once (bounded batch, O_DIRECT or io_uring) before CPU compute starts, so queue depth > 1.
  (3) Group prefill rows per expert (check what `-ub` splitting does today: each ubatch reloads).
- Test: Qwen only (final run); decode tok/s and MB/s paired against EXPLICIT_READ off/on.
- Compression (their rANS, -33% SSD traffic): TESTED for compressibility on our tensors, not built. Field-wise ideal entropy coding of Qwen's expert blocks saves 2.7% (IQ3_S gate/up,
  110 B blocks: scale 61%, grid index 98%, high bits 95%, signs 100%, 4-bit scales 81% of raw) and 5.1% (IQ4_NL down: nibbles 3.91 of 4 bits); lzma finds 0% and 1%. A GPU decoder for <= 5% less SSD traffic is
  not worth building unless the quant changes (a GSQ-RCO file with different types should be re-measured with the same script before this is closed).

## L6. Smaller items

- HOBBIT's layer-aware eviction term (FLD, farthest layer distance) and "high-precision frequency" (LHU): reported 4.7-8.7% fewer miss-penalty vs LRU. A layer-distance term in `moe_cache_weighted_heat` is a one-line experiment.
- vLLM/vAttention reclaim: implemented as `ggml_backend_cuda_moe_cache_yield_vram` (see ROADMAP step 1).

## Order of work (gemma-testable first)

L1 -> L2 -> L3 -> L4 -> then L5 measurement during the final Qwen run. Each is a flag-gated variant, off until it wins paired.

## Correction recorded here

The Qwen file we run (UD-IQ4_XS) already stores the routed experts mostly as IQ3_S/IQ4_NL, so the GSQ-RCO IQ3_S file will cut bytes per expert by less than the 27% file-size
difference suggests (its saving is largely outside the expert tensors). The expected speedup from it is a hypothesis for the final run, not an estimate to rely on.
