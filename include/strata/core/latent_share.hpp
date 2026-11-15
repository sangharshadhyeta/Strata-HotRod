// Adapted from Edge0's edge0-strata (https://github.com/Edge0-AI/Edge0, MIT licence: a fork of Niko1221's Strata), branch hotrod-edge0. Taken as it is.
// Latent MoE: the one session-level D/U device pair, stored by the engine's LatentTab and pulled by the
// prefill and verify arms inside their own init (no per-object external wiring to miss).  A non-latent pack
// never stores a pair, so both accessors stay null.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata {
namespace core {

inline uint8_t*& latent_shared_d() { static uint8_t* p = nullptr; return p; }
inline uint8_t*& latent_shared_u() { static uint8_t* p = nullptr; return p; }

inline const uint8_t*& latent_xl_current() { static const uint8_t* p = nullptr; return p; }
inline const uint32_t*& latent_xlseq_current() { static const uint32_t* p = nullptr; return p; }
inline const uint8_t*& latent_act0_current() { static const uint8_t* p = nullptr; return p; }

// (ours, not Edge0's) what the verify and prompt paths need besides the D/U pointers, all read off the pack by LatentTab: the group count,
// the D/U format, one layer's D and U bytes, and the expert group labels ([n_layers * n_expert], on the device and on the host).
struct LatentShared {
    int groups = 0;
    int fmt = 8;                      ///< 8 = Q8_0, 4 = Q4_0
    size_t d_layer_bytes = 0, u_layer_bytes = 0;
    const int32_t* labels_dev = nullptr;
    const int32_t* labels_host = nullptr;
};
inline LatentShared& latent_shared_info() { static LatentShared s; return s; }

}  // namespace core
}  // namespace strata
