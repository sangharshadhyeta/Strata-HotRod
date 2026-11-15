// Adapted from Edge0's edge0-strata (https://github.com/Edge0-AI/Edge0, MIT licence: a fork of Niko1221's Strata), branch edge0-latent. Taken as it is.
// Latent MoE: the one session-level D/U device pair, stored by the engine's LatentTab and pulled by the
// prefill and verify arms inside their own init (no per-object external wiring to miss).  A non-latent pack
// never stores a pair, so both accessors stay null.
#pragma once
#include <cstdint>

namespace strata {
namespace core {

inline uint8_t*& latent_shared_d() { static uint8_t* p = nullptr; return p; }
inline uint8_t*& latent_shared_u() { static uint8_t* p = nullptr; return p; }

inline const uint8_t*& latent_xl_current() { static const uint8_t* p = nullptr; return p; }
inline const uint32_t*& latent_xlseq_current() { static const uint32_t* p = nullptr; return p; }
inline const uint8_t*& latent_act0_current() { static const uint8_t* p = nullptr; return p; }

}  // namespace core
}  // namespace strata
