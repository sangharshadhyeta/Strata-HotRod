#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <string>

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
    float wmax;   ///< HOBBIT-style weight floor (STRATA_ROUTE_WMAX): a missed pick is replaced only when the router's own weight on it is below this (0 = off, the rank floor alone decides)
    float budget; ///< STRATA_ROUTE_BUDGET=<b>: ranks below `lo` take the budget test w * gap <= b instead of the margin / edge (0 = off)
    float budget_tail; ///< STRATA_ROUTE_BUDGET_TAIL=<b>: ranks at or above `lo` take a resident stand-in only when w * max(gap, 0) <= b (0 = off: any, as the margin says)
};
void route_cfg_init(int lo, int hi, float margin, float edge, float bias = 0.f, int maxrank = 0, float want_lo = 1.f, float repl_gap = 0.f, float wmax = 0.f, float budget = 0.f, float budget_tail = 0.f);   ///< before graph capture
void route_cfg_set_floor(int lo);                                ///< between windows
void route_cfg_set_budget(float budget);                         ///< between windows (STRATA_ROUTE_BUDGET_DYN)
void route_cfg_set_budgets(float budget, float budget_tail);     ///< between windows (STRATA_ROUTE_BUDGET_AUTO): both budgets at once
const RouteCfg* route_cfg_device();
/// The same contract for a model family's (experts, top-k): 512 / 10 (Flash-Next: the two calls above, unchanged) and
/// 256 / 8 (Qwen3.6-35B-A3B).  logits [n_tok, n_expert], ids / weights [n_tok, k].  CUDA / HIP.
bool native_router_topk_supported(int n_expert, int k);
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream);
/// STRATA_ROUTE_RESIDENT: swap non-resident picks of the segment's floor ranks for resident ones within the margin (see the .cu).
/// `stats` is nullptr or device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after; with `cheap` set, [4] the
/// rank 0-1 stand-ins made for an expert outside the RAM tier (STRATA_ROUTE_RAM_FLOOR), so a block of 5 or more.
/// `want` is nullptr or the layer's slice of the wanted-expert counter (route_want_layer): the router's own top-10, counted
/// before any swap. Needs route_cfg_init first.
/// `rows` is nullptr or the layer's [8 rows][10] slice of the per-row record (route_rows_layer): the same original picks per window row.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           unsigned long long* stats, void* stream, float* want = nullptr, int32_t* rows = nullptr, float* repl = nullptr,
                           const uint8_t* ring = nullptr, const uint8_t* cheap = nullptr, unsigned long long* bstats = nullptr,
                           float* row_shift = nullptr);
/// STRATA_FRAGILE_RECHECK: nullptr or n_tok floats, zeroed by the caller before the window; each replaced pick adds its router
/// weight times max(gap, 0) to its row (gap: the pick's logit minus the stand-in's).  Off: nullptr, and the kernel adds nothing.
/// STRATA_RING (the look-ahead ring): one byte per (layer, expert) on the device, 1 while a Foresight ring slot holds the expert and
/// its copy has landed (set on the copy stream after the copy, cleared before the slot is refilled).  The router counts such an
/// expert as resident: a pick it holds is computed exactly on the GPU from the slot, and it may serve as a stand-in.  Allocated
/// once, outside graph capture, before the first window.
/// STRATA_RING_GPU (the look-ahead as a graph node): at each layer l of a decode window the router of layers l+1 and l+2 is applied to
/// layer l's MoE input on the GPU and their top-10 are written straight into host-mapped memory, [n_layers][2][8 rows][10], where the
/// pool reads them at layer l's dispatch (after the doorbell, which the kernels precede in the stream).  Punctual: a fixed lead of one
/// and two layers, no thread to be busy or late.  Allocated once, outside graph capture.
void route_pred_alloc(int64_t n_layers);
bool route_pred_active();
const int32_t* route_pred_host();                                   ///< the table the host reads
int32_t* route_pred_dev(int64_t target_layer, int dist);            ///< dist 1 or 2: the slab of 8 x 10 ids, device alias
float* route_pred_logits();                                          ///< scratch, 8 x 512 floats
float* route_pred_weights();                                         ///< scratch, 8 x 10 floats
/// STRATA_LA_MODEL=<file> (opt-in): the published per-layer d=1 linear expert predictors, folded into one bf16 [512 x n_embd]
/// matrix and a bias per target layer (tools/predictor/export_la_model.py writes the file).  Loaded once before graph capture;
/// false (with the reason in `msg`) leaves the look-ahead on the routers.  The decode look-ahead's next-layer guess then applies
/// the target layer's matrix and bias instead of that layer's router.
bool route_pred_model_load(const char* path, int64_t n_layers, int64_t n_embd, std::string& msg);
const uint16_t* route_pred_model_w(int64_t target_layer);           ///< null: no predictor for this layer (the router is used)
const float* route_pred_model_b(int64_t target_layer);
void route_pred_bias_add(float* logits, const float* bias, int n_tok, void* stream);   ///< logits [n_tok x 512] += bias
/// STRATA_LA_EDGE_SKIP (default on, =0 off): route_pred_edge_skip_alloc once before graph capture; route_pred_edge_skip after the look-ahead's top-10
/// writes a predicted rank lo-1 id as -1 when a resident stand-in is within the edge (the floor's config); skips are counted on the
/// device, and route_pred_edge_skips_take returns the count since the last take and zeroes it.
void route_pred_edge_skip_alloc();
void route_pred_edge_skip(const float* pred_logits, int32_t* pred_ids, const int32_t* res_layer, int n_tok, void* stream);
unsigned long long route_pred_edge_skips_take();
/// STRATA_LA_TAIL_READ (opt-in): route_pred_tail_read after the look-ahead's top-10 of a layer (its weights in route_pred_weights, the logits in
/// route_pred_logits) writes a predicted rank lo..hi id as -1 when the router would replace it with a resident stand-in (the window's budgets).
/// route_pred_tail_read_take returns the count since the last take and zeroes it.  route_cfg_tail_open: the open rule (no tail reads needed).
void route_pred_tail_read_alloc();
void route_pred_tail_read(const int32_t* res_layer, int32_t* pred_ids, int n_tok, void* stream);
unsigned long long route_pred_tail_read_take();
bool route_cfg_tail_open();
void route_ring_alloc(size_t n);
uint8_t* route_ring_ptr();                                   ///< the whole table (nullptr when not allocated)
const uint8_t* route_ring_layer(int64_t layer, int64_t n_expert);   ///< the layer's slice, nullptr when not allocated
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
/// STRATA_ROUTE_RAM_FLOOR=1 (opt-in, read once): a non-resident rank 1 pick (rank 0 too with STRATA_ROUTE_RAM_FLOOR_RANK0=1) may
/// take a stand-in within STRATA_ROUTE_RAM_FLOOR_MARGIN logits (`cheap`, below, marks the RAM tier).  Measured 20 Oct (210 MMLU-Pro
/// questions): 26% of decode time was the CPU path waiting on SSD fetches for such picks.  Off again by default (20 Oct night):
/// broken answers in a pi harness session on a cold card (hit rate 59-85%), cause not yet separated from the ranks 2-9 stand-ins
inline bool route_ram_floor_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ROUTE_RAM_FLOOR"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}
/// The RAM-tier table of STRATA_ROUTE_RAM_FLOOR: one byte per (layer, expert), 1 while the expert has a RAM-tier place.  Host-mapped
/// (the host writes it, the router reads it at each launch).  Allocated once, outside graph capture, before the complement is ready.
void route_cheap_alloc(size_t n);
uint8_t* route_cheap_host();                                        ///< the host copy (nullptr when not allocated)
const uint8_t* route_cheap_layer(int64_t layer, int64_t n_expert);  ///< the layer's device slice (nullptr when not allocated)
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
