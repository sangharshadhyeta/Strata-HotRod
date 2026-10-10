#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// STRATA_ROUTE_RESIDENT: swap non-resident tail-rank experts for resident ones within `margin` logits (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
/// `want` is nullptr or the layer's slice of the wanted-expert counter (route_want_layer): the router's own top-10, counted
/// before any swap.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream, float* want = nullptr);
/// The wanted-expert counter: what the unmasked router picked, for the swap round (a router confined to resident experts
/// cannot tell the swap round which expert to bring in). Allocated once (outside graph capture), n_layers * n_expert floats.
void route_want_alloc(size_t n);
float* route_want_layer(int64_t layer, int64_t n_expert);   ///< nullptr when not allocated
bool route_want_active();
/// Adds the counts since the last call to the host `usage` table (n_layers * n_expert) and clears them.
void route_want_drain(float* usage, size_t n);
/// STRATA_TRACE_ADAPT: cumulative counters of the decode rows' router picks (16 uint64): [0..9] picks of rank r that were not
/// resident, [10] router weight on them (x1e6), [11] rows, [12] rows with the top-1 not resident, [13] rows with a
/// non-resident pick among the top 5. Allocated outside graph capture; read on the host between windows.
void route_detail_alloc();
void route_detail_read(unsigned long long* out16);   // 48 entries: [16..21] replaced-pick weight buckets (<1%,<2,<5,<10,<20,>=20), [22..26] substitute rank in the router ranking (<20,<40,<80,<160,>=160), [27..32] score gap (<0.5,<1,<2,<4,<8,>=8), [33..42] summed weight per rank x1e6
}
