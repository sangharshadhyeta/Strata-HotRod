// include/strata/core/peer_retry.hpp - the peer tier's reserve, raised by what its prompt buffers were short (HotRod, 21 Nov 2026).
//
// WHY.  --peer-reserve-mib (default 600) is what the peer card keeps free after its expert cache is filled; the prompt path's buffers are allocated
// from it afterwards, and their size depends on the chunk and --peer-prefill-rows.  On two RTX 3060 12 GB cards (community report, issue #2) 600 filled
// the cache and left the buffers short, so every prompt row stayed on the primary card; 1200 worked.  The engine cannot know the size before it has
// allocated them, but it can read how short the first failing allocation was, close the peer cache, reopen it with a larger reserve and try once more.
//
// The rule: each attempt adds the shortfall plus a margin (the allocations after the failing one are not counted, so the margin and the attempts cover
// them), never less than 256 MiB, and the reserve never goes past a cap, so the retry cannot eat the whole card's cache.  Three attempts at most.
//
// EXPERIMENTAL: written without a second GPU to run it on (the rule is unit-tested; the reopen is not).  STRATA_PEER_RESERVE_RETRY=0 turns it off.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

constexpr int kPeerRetryMaxAttempts = 3;
constexpr int kPeerRetryCapMiB = 4096;     ///< the reserve never exceeds this
constexpr int kPeerRetryMinStepMiB = 256;
constexpr int kPeerRetryMarginMiB = 256;

/// The reserve to try next after `attempt` (1-based) failed with `reserve_mib` and a first failing allocation that was `missing_mib` short.
/// Returns `reserve_mib` unchanged when the cap is already reached (no further attempt is worth making).
inline int next_peer_reserve_mib(int reserve_mib, int64_t missing_mib, int attempt) {
    const int64_t step = std::max<int64_t>(kPeerRetryMinStepMiB, std::max<int64_t>(0, missing_mib) + kPeerRetryMarginMiB);
    const int64_t next = (int64_t) reserve_mib + step * std::max(1, attempt);
    return (int) std::min<int64_t>(next, std::max(reserve_mib, kPeerRetryCapMiB));
}

}  // namespace strata::core
