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
/// The stand-in floors (SPEC 17), in host-mapped memory the router kernel reads at every launch, so a captured graph follows a
/// change the host makes between windows. `lo[0..2]` are the floors of thinking, of the answer, and of code (the first rank that
/// may be replaced; a HIGHER number replaces less, so it is the more careful); `eff` is the floor in force now, which the kernel
/// reads: the floor of the part being written, and inside a code fence never less careful than the code floor. `hi` the last
/// rank (9); `margin` the largest score gap a replacement may have (1e30 = any); `edge` >= 0 makes the floor fractional: the
/// rank below the floor is replaced too, but only by a substitute within `edge` logits of it; -1 = off.
struct RouteCfg {
    int32_t lo[3];
    int32_t hi;
    float margin;
    float edge;
    int32_t eff;
};
void route_cfg_init(int lo_think, int lo_prose, int lo_code, int hi, float margin, float edge);   ///< before graph capture
/// The state of the answer being written: inside the thinking, inside a code fence. Never lowers the floor below the part's own.
void route_cfg_set_state(bool in_think, bool in_fence);
/// A request's floors (-1 = the environment's) and whether it starts inside the thinking; a no-op while the stand-in is off.
void route_cfg_request(int lo_think, int lo_prose, int lo_code, bool in_think);
const RouteCfg* route_cfg_device();
/// STRATA_ROUTE_RESIDENT: swap non-resident picks of the segment's floor ranks for resident ones within the margin (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
/// `want` is nullptr or the layer's slice of the wanted-expert counter (route_want_layer): the router's own top-10, counted
/// before any swap. Needs route_cfg_init first.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           unsigned long long* stats, void* stream, float* want = nullptr);
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
