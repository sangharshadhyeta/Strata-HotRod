// include/strata/core/latent_tab.hpp - the session-level latent MoE tables (branch edge0-latent).
//
// An Edge0 latent pack's experts run in a latent space: every layer has D projections (hidden -> latent, one per expert group), U
// projections (latent -> hidden) and a group label per expert.  This loads them ONCE for the session, to the GPU, before the expert cache is
// sized (so the cache's "what is free" already excludes them), and publishes the device pointers through latent_share.hpp for the verify and
// prompt paths to pull inside their own init.
//
// Nothing here is tied to a model or a card.  Everything is read off the pack: the latent width and the hidden width from the layout, the
// group count and the D/U format (Q8_0 or Q4_0) from the D/U file's SIZE, the labels from labels.bin.  Only one decision is a policy, and it
// has a flag: when both a Q8 and a Q4 D/U file are beside the pack, the Q8 one is used while it takes at most STRATA_LATENT_DU_SHARE_PCT
// (default 50) percent of the free VRAM, else the Q4 one.  STRATA_LATENT_DENSE (a file) and STRATA_LATENT_DENSE_FMT (4 | 8) force the choice.
#pragma once

#include "strata/core/latent_dense.hpp"
#include "strata/core/latent_share.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// What a D/U file is, from its size alone: its format (8 = Q8_0, 4 = Q4_0) and the number of groups, such that
/// file_bytes == n_layers * groups * (latent * (n_embd/32) + n_embd * (latent/32)) * block_bytes.  False when no format and group count
/// (1..64) fit, or when both do (the file cannot be told apart).
bool latent_dense_identify(uint64_t file_bytes, int64_t n_layers, int64_t n_embd, int64_t latent, int& fmt, int& groups);

struct LatentDenseChoice {
    std::string path;
    int fmt = 0;           ///< 8 or 4
    int groups = 0;
    uint64_t bytes = 0;    ///< the whole file: every layer's D and U
};

/// Picks the D/U file for a pack (see the header comment).  `free_vram` is what the GPU has free now; the choice fails when the file it picks
/// does not fit in it.  `share_pct` <= 0 reads STRATA_LATENT_DU_SHARE_PCT (default 50).
bool latent_dense_choose(const std::string& pack_dir, int64_t n_layers, int64_t n_embd, int64_t latent, uint64_t free_vram, int share_pct,
                         LatentDenseChoice& out, std::string& err);

class LatentTab {
public:
    LatentTab() = default;
    ~LatentTab();
    LatentTab(const LatentTab&) = delete;
    LatentTab& operator=(const LatentTab&) = delete;

    /// Loads (once; a second call only republishes the pointers).  `pack_dir` holds labels.bin and the latent_dense_q*.bin files.
    bool load(const std::string& pack_dir, int64_t n_layers, int64_t n_embd, int64_t latent, int64_t n_expert, uint64_t free_vram,
              std::string& err);
    bool loaded() const { return d_ != nullptr; }
    int groups() const { return groups_; }
    int fmt() const { return fmt_; }
    const LatentDense& dense() const { return dense_; }
    const std::vector<int32_t>& labels() const { return labels_; }   ///< [n_layers * n_expert]
    const uint8_t* d() const { return d_; }                          ///< device: every layer's D, then (u) every layer's U
    const uint8_t* u() const { return u_; }
    uint64_t device_bytes() const { return device_bytes_; }

private:
    LatentDense dense_;
    std::vector<int32_t> labels_;
    uint8_t* d_ = nullptr;
    uint8_t* u_ = nullptr;
    int groups_ = 0, fmt_ = 0;
    uint64_t device_bytes_ = 0;
};

}  // namespace strata::core
