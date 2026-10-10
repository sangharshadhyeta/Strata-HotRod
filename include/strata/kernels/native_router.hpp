#pragma once
#include <algorithm>
#include <cstdlib>
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
/// The stand-in's ONE floor, in host-mapped memory the router kernel reads at every launch, so a captured graph follows a change
/// the host makes between windows (nothing moves it today; a controller that follows the drive's load, SPEC 19.8, would).
/// `lo` is the first rank that may be replaced (a HIGHER number replaces less, so it is the more careful); `hi` the last (9);
/// `margin` the largest score gap a replacement may have (1e30 = any); `edge` >= 0 makes the floor fractional: the rank below the
/// floor is replaced too, but only by a substitute within `edge` logits of it; -1 = off. (SPEC 17's per-part floors, with a
/// thinking / fence tracker, were built, measured and REMOVED: no gain from a higher code floor, see SPEC 17 and 18.1.)
struct RouteCfg {
    int32_t lo;
    int32_t hi;
    float margin;
    float edge;
    float repl_gap;    ///< SWAP2: only a replaced pick whose stand-in is at least this many logits below it is counted for a quality swap (0 = every one)
    float want_lo;     ///< the demand counter's weight for a pick the floor replaces (rank >= lo); a pick below the floor counts 1 (default 1)
    int32_t maxrank;   ///< a stand-in must sit within this many places of the router's own ranking of all 512 experts (0 = any)
    float bias;   ///< CCE router bias: ln(1 + beta) added to a resident expert's logit before the top-10 (0 = off)
};
void route_cfg_init(int lo, int hi, float margin, float edge, float bias = 0.f, int maxrank = 0, float want_lo = 1.f, float repl_gap = 0.f);   ///< before graph capture
void route_cfg_set_floor(int lo);                                ///< between windows
const RouteCfg* route_cfg_device();
/// STRATA_ROUTE_RESIDENT: swap non-resident picks of the segment's floor ranks for resident ones within the margin (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
/// `want` is nullptr or the layer's slice of the wanted-expert counter (route_want_layer): the router's own top-10, counted
/// before any swap. Needs route_cfg_init first.
/// `rows` is nullptr or the layer's [8 rows][10] slice of the per-row record (route_rows_layer): the same original picks per window row.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           unsigned long long* stats, void* stream, float* want = nullptr, int32_t* rows = nullptr, float* repl = nullptr);
/// SWAP2 (STRATA_SWAP2_MAX > 0): the experts stand-ins replaced, counted per (layer, expert) on the device (route_repl_layer is the layer's slice);
/// route_repl_drain adds them to a host array and clears them.  The swap round may bring the most replaced ones into slots nothing needs.
inline int route_swap2_max() {
    static const int v = [] { const char* e = std::getenv("STRATA_SWAP2_MAX"); return e != nullptr ? std::max(0, std::atoi(e)) : 2; }();
    return v;
}
inline float route_swap2_gap() {
    static const float v = [] { const char* e = std::getenv("STRATA_SWAP2_GAP"); return e != nullptr ? std::max(0.0f, (float) std::atof(e)) : 1.0f; }();
    return v;
}
void route_repl_alloc(size_t n);
float* route_repl_layer(int64_t layer, int64_t n_expert);
bool route_repl_active();
void route_repl_drain(float* usage2, size_t n);
/// SPEC 13.40 (STRATA_COUNT_KEPT=1): the per-row record of the router's original top-10, so the swap round counts only the rows of a
/// window whose input token was kept (rows 0..a after a accepted drafts).  `route_rows_apply(kept, ...)` adds those rows' picks to the
/// host usage table after the acceptance is known; with this on the atomic count (`want`) is not used.
void route_rows_alloc(int64_t n_layers);
int32_t* route_rows_layer(int64_t layer);   ///< nullptr when not allocated
bool route_rows_active();
void route_rows_apply(int kept, float* usage, int64_t n_layers, int64_t n_expert);
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
