// src/drafter/dflash2.hpp - the native DFlash 2 drafter (Qwen3.8-Flash-Next-DFlash2), in the engine (22 Oct 2026).
//
// Mirrors z-lab's DFlash2DraftModel (dflash/model.py) at batch 1, greedy (temperature 0):
//   * feed():    the context features of positions [ctx_lo, ctx_hi), as the trace and the engine hold them (bf16, 5 traced
//                layers x 2560 = 12800 per position).  Each position goes through fc and hidden_norm once, then through every
//                decoder layer's k/v projections, k_norm and rope into a 2048-slot ring per layer (the sliding window is 2048).
//   * propose(): the block of noise rows (the anchor, then mask tokens) through the five layers, each with its two-tap
//                dynamic causal convs and the sliding attention over the ring and the block's own keys; then the selector
//                picks one path through the top-16 candidates of the head's logits.  -> block - 1 draft tokens.
// The token embedding and the output head are the target's: the caller passes them as callbacks (run on the engine's stream).
// Weights come from the GGUF (Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf or -Q8_0.gguf): the matrices through native_mmvq
// (Q4_K / Q6_K in the Q4_K_M file, Q8_0 in the Q8_0 file), the norms and conv bases as f32, the selector codebooks as
// Q4_K or Q8_0 rows (one row = 256 values; both codebooks have the same type).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::drafter {

// Token ids in, fp32 rows out: out[i] = embed(ids[i]) for n tokens; ids and out are device memory.
using EmbedFn = std::function<bool(const int32_t* ids_dev, int n, float* out_dev, void* stream, std::string& err)>;
// Hidden rows in, logits out: logits[r * vocab + v] for n rows of hidden (fp32, device memory).
using HeadFn = std::function<bool(const float* hidden_dev, int n, float* logits_dev, void* stream, std::string& err)>;

// What a feed or a propose computed, copied to the host.  Filled only when a pointer is passed (the reference test).
struct Dflash2Trace {
    std::vector<float> th_norm;                 // feed: hidden_norm(fc(features)) of the fed rows, [rows, 2560]
    std::vector<std::vector<float>> layer;      // propose: each decoder layer's output over the block's rows, [block, 2560]
    std::vector<float> final;                   // propose: norm(last layer) over rows 1.., [block - 1, 2560]
    std::vector<int32_t> cand;                  // propose: the selector's top-16 candidates per row, [block - 1, 16]
    std::vector<float> cand_logit;              // propose: their head logits, [block - 1, 16]
};

class Dflash2 {
public:
    static constexpr int kHidden = 2560;
    static constexpr int kLayers = 5;
    static constexpr int kHeads = 32;
    static constexpr int kKvHeads = 8;
    static constexpr int kHeadDim = 128;
    static constexpr int kFfn = 7680;
    static constexpr int kWindow = 2048;        // sliding_window: a query sees keys within 2047 positions either side
    static constexpr int kBlock = 8;            // the largest block (the mmvq takes up to 8 columns)
    static constexpr int kVocab = 248320;
    static constexpr int kRank = 256;           // selector_rank
    static constexpr int kTopK = 16;            // selector_top_k
    static constexpr int kCtxFeat = 12800;      // 5 traced layers x kHidden
    static constexpr int kMaskId = 248077;      // the drafter's mask token (tokenizer.ggml.mask_token_id)

    Dflash2();
    ~Dflash2();
    Dflash2(const Dflash2&) = delete;
    Dflash2& operator=(const Dflash2&) = delete;

    // Loads the drafter's tensors from the GGUF onto the current device.  False with err on any missing or mistyped tensor.
    bool load(const std::string& gguf_path, std::string& err);
    void unload();
    bool loaded() const;

    // A new request: the context and the draft cache are emptied.
    void reset();

    // The features of positions [ctx_lo, ctx_hi), bf16 bits on the device, row-major [ctx_hi - ctx_lo, kCtxFeat].
    // ctx_lo must be the position after the last fed one.  Returns false with err when it is not.
    bool feed(const uint16_t* feats_dev, int ctx_lo, int ctx_hi, void* stream, std::string& err,
              Dflash2Trace* trace = nullptr);

    // The block for the anchor token at position `start` (rows start..start+block-1: the anchor, then mask tokens).
    // Needs the features of every position < start fed, and none at or after start (the ring is the window).  draft gets
    // block - 1 tokens: the selector's path.  Fails with err when start is not after the last start.
    bool propose(int start, int32_t anchor, int block, const EmbedFn& embed, const HeadFn& head,
                 std::vector<int32_t>& draft, void* stream, std::string& err, Dflash2Trace* trace = nullptr);

    int fed() const;          // positions with features (the next ctx_lo)
    int last_start() const;   // the last propose's start (0 after reset)

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::drafter
