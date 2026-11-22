// include/strata/inkling2/drafter.hpp - Inkling 2's DSpark drafter (the official checkpoint mtp.safetensors) for speculative decoding.
//
// Written from the design in Project Maya's maya-ik2 (src/ik2/ik2_draft.hpp / .cu, MIT; credit in third_party/project-maya/NOTICE.md) and the checkpoint's own tensors; the kernels are ours.
//   * The target leaves 4 hidden states per position (Mini): its stream at the start of visits 5, 23 and 41, and the final normed hidden ("taps").  For each committed position each draft layer
//     mixes them (softmax(fuse_weight[layer])), RMS-norms (its dflare_norm), projects K / V (k_proj_ctx, v_proj_ctx; per-head norm, NeoX RoPE) into its own context cache.  The last committed
//     position's mix (fuse_weight[last], the Markov head's dflare_norm) is f_rnn.
//   * A block of 8 rows: the anchor (the last token the target chose: its embedding as the target embeds) then 7 mask_embed rows, through 3 dense layers (q / k norm, NeoX RoPE theta 1e6,
//     SiLU MLP; layers 0 and 2 see the last 1024 context positions, layer 1 all of it; every row sees the block), the final norm, the target's output head for the base logits.
//   * The Markov head drafts the 7 tokens one by one: a GRU-like cell on [state; markov_w1[prev]; hidden; f_rnn] gives a latent whose bias on the logits is (head . hidden_out) latent + head . b,
//     folded into a [R][vocab] matrix once; the token is the argmax of base + bias.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace strata::inkling2 {

class Drafter {
public:
    /// Loads the checkpoint onto the current device.  C: the target's width (2304 = Mini); ctx: positions the context caches hold; stream: every launch.  Throws std::runtime_error.
    Drafter(const std::string& path, int C, int ctx, void* stream);
    ~Drafter();
    Drafter(const Drafter&) = delete;
    Drafter& operator=(const Drafter&) = delete;

    /// The target's head as Q8_0 rows (device, row_bytes each, n_rows of them; n_vocab real tokens among them): folds the Markov head through it (once, a few seconds).
    void fold(const uint8_t* head_q8, int n_rows, int n_vocab);
    /// The context cache rows for positions pos0 .. pos0 + m - 1 from tap rows r0 .. r0 + m - 1 of taps [n_taps][ld][C] (device, F32).
    void write_context(const float* taps, int ld, int r0, int m, int pos0);
    /// f_rnn from tap row `row`.
    void set_f_rnn(const float* taps, int ld, int row);
    /// hidden [rows][C] (device) -> logits [rows][n_rows] (device) through the target's head.
    using HeadFn = std::function<void(const float* hidden, int rows, float* logits)>;
    /// `proposals` (<= 7) tokens drafted after `anchor` (the last token the target chose, at position P; anchor_h its embedding as the target embeds, device [C]).
    /// probs (optional): each drafted token's probability, as the Qwen draft layer reports it, for the min-p rule.
    void propose(int anchor, const float* anchor_h, int P, int proposals, const HeadFn& head, int* out, float* probs = nullptr);
    int n_taps() const;
    int n_rows() const;
    size_t bytes() const;
    /// Visits whose input stream the target must keep as taps 0 .. n_taps - 2 (the last tap is the final normed hidden).
    static const int* tap_visits(int& n);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::inkling2
