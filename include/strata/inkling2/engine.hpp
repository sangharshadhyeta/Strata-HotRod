// include/strata/inkling2/engine.hpp - the Inkling 2 GPU engine, F2 of docs/INKLING-PORT.md: the whole looped forward on the GPU, one token a call.
//
// Built from what the engine already has (Q8_0 / NVFP4 mat-vecs through iq_mmvq, the short-conv kernel, RMSNorm, the DeepSeek-derived router) and the new
// kernels of inkling2_kernels.hpp; float32 activations.  Dense weights live in VRAM; the routed experts are read where they are, zero-copy over PCIe, from the
// GGUF mapped and registered with CUDA.  That access sits behind `ExpertPtrs expert(layer, e)`, the one seam F3 replaces with the RAM-tier ExpertStore and the
// VRAM slot cache.  Positions are teacher-forced or decoded alike: `step(token, pos)` runs all 48 visits for one token against the per-visit KV and conv state.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/inkling2.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace strata {
namespace inkling2 {

/// Where an expert's three matrices are (device-addressable pointers) and its scales (NVFP4: the per-expert F32 scale, else 1).
struct ExpertPtrs {
    const void *gate = nullptr, *up = nullptr, *down = nullptr;
    int type = 0;                  ///< ggml type of the matrices (40 NVFP4, 8 Q8_0)
    float gate_up_scale = 1.0f, down_scale = 1.0f;
};

/// Sees the seams of a visit as they are produced (device pointers, n floats): the names are the reference's (k_conv, v_conv, q, k, v, r, attn_gated,
/// attn_out, ffn_out, out, loop_q_in, loop_ffn_in).
using SeamSink = std::function<void(int visit, const char* name, const float* dev, int n)>;

class Engine {
public:
    /// Uploads the dense weights and registers the experts.  max_ctx: positions of the per-visit KV caches.
    Engine(const GgufFile& g, const Config& c, const Layout& lay, int max_ctx);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// One token at position `pos` (0, 1, 2 ... in order): all visits; fills `logits` (the real vocabulary, softcapped) when not null.
    void step(int64_t token, int pos, std::vector<float>* logits, const SeamSink& sink = nullptr);
    /// Forgets the conversation (conv states and positions); the KV is simply overwritten.
    void reset();
    /// The picks of the last step, per visit: [visit][k] expert ids (for tests and the learners).
    const std::vector<std::vector<int>>& last_picks() const;
    /// The routed experts' bytes and the dense weights' bytes in VRAM.
    uint64_t expert_bytes() const;
    uint64_t dense_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace inkling2
}  // namespace strata
