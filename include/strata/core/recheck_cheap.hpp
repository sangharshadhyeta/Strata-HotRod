// include/strata/core/recheck_cheap.hpp - the cheap fragile re-check (docs/SPEC-recheck-cheap.md, changes 1 and 2), header-only, no CUDA.
//
// Change 1: the exact successor of a cut makes only its row 0 exact. The fragile control flag is a row count n: rows < n keep the
// router's picks (no stand-in move, no mask move); n = 0 is the normal path; n >= T is the old control window. The successor sets
// n = 1 and keeps the request's budget, unless STRATA_RECHECK_ROW_ONLY=0 (the whole-window exact successor of before).
//
// Change 2: the cut row's replaced picks (the mask-learn event list of that row) are queued for a read ahead of the successor, deduplicated,
// capped, layer-ordered (the successor's layers run in ascending order), and the ones already resident are skipped.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace strata::core::recheck {

/// The row count that means "every row" (any window has fewer rows): the fragile control window's flag.
constexpr int kAllRows = 1 << 30;

/// The exact successor's plan. row_only (STRATA_RECHECK_ROW_ONLY on and the flag present) keeps the request's budget and the
/// window's normal exact/unmasked state for the other rows; it sets no win_exact and no zero budget. Otherwise the successor is the
/// whole-window exact re-run of before: budget 0, tail 1e-30, win_exact (no mask moves).
struct SuccessorPlan {
    bool row_only;      ///< the flag row count is 1 for this window
    bool win_exact;     ///< the whole window is an exact re-run (no mask moves, no mask learning exposure)
    bool zero_budget;   ///< the budget is 0 and the tail 1e-30 (every rank exact)
};
inline SuccessorPlan successor_plan(bool successor, bool row_only) {
    SuccessorPlan p{};
    p.row_only = successor && row_only;
    p.win_exact = successor && !row_only;
    p.zero_budget = successor && !row_only;
    return p;
}

/// One replaced pick of a cut row: its layer and the expert it replaced.
struct Pick {
    int32_t layer;
    int32_t expert;
    int32_t rank = 0;   ///< the replaced pick's router rank (0-9); used by plan_prefetch_ranked only
    float risk = 0.f;   ///< the event's risk (w x gap); used by plan_prefetch_ranked only
};

/// What the prefetch does with a cut row's picks: `todo` in queue order (ascending layer, first appearance within a layer), at most
/// `cap` of them; `resident` the picks skipped as already resident; `duplicate` the repeats of a pick already seen; `over_cap` the
/// fresh, non-resident picks past the cap (dropped).
struct PrefetchPlan {
    std::vector<Pick> todo;
    size_t resident = 0;
    size_t duplicate = 0;
    size_t over_cap = 0;
};

template <class IsResident>
inline PrefetchPlan plan_prefetch(const std::vector<Pick>& picks, size_t cap, IsResident is_resident) {
    PrefetchPlan p;
    std::set<std::pair<int32_t, int32_t>> seen;
    std::vector<Pick> fresh;
    for (const Pick& k : picks) {
        if (!seen.insert({k.layer, k.expert}).second) {
            ++p.duplicate;
            continue;
        }
        if (is_resident(k.layer, k.expert)) {
            ++p.resident;
            continue;
        }
        fresh.push_back(k);
    }
    std::stable_sort(fresh.begin(), fresh.end(), [](const Pick& a, const Pick& b) { return a.layer < b.layer; });
    if (fresh.size() > cap) {
        p.over_cap = fresh.size() - cap;
        fresh.resize(cap);
    }
    p.todo = std::move(fresh);
    return p;
}

/// SPEC-recheck-gated (E): the cut row's picks ranked for the prefetch.  Order: rank ascending, then risk descending (a NaN risk is 0), then
/// layer and expert ascending (a fixed order).  A repeat keeps its first (best-ranked) place; a resident pick is skipped and counted; the first
/// `cap` fresh picks in that order are taken, the rest of the fresh ones count as over_cap.  The queue order is ascending layer (stable).
template <class IsResident>
inline PrefetchPlan plan_prefetch_ranked(std::vector<Pick> picks, size_t cap, IsResident is_resident) {
    auto risk_of = [](const Pick& k) { return k.risk == k.risk ? k.risk : 0.0f; };
    std::stable_sort(picks.begin(), picks.end(), [&](const Pick& a, const Pick& b) {
        if (a.rank != b.rank) return a.rank < b.rank;
        if (risk_of(a) != risk_of(b)) return risk_of(a) > risk_of(b);
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.expert < b.expert;
    });
    PrefetchPlan p;
    std::set<std::pair<int32_t, int32_t>> seen;
    std::vector<Pick> chosen;
    for (const Pick& k : picks) {
        if (!seen.insert({k.layer, k.expert}).second) {
            ++p.duplicate;
            continue;
        }
        if (is_resident(k.layer, k.expert)) {
            ++p.resident;
            continue;
        }
        if (chosen.size() < cap) chosen.push_back(k);
        else ++p.over_cap;
    }
    std::stable_sort(chosen.begin(), chosen.end(), [](const Pick& a, const Pick& b) { return a.layer < b.layer; });
    p.todo = std::move(chosen);
    return p;
}

}  // namespace strata::core::recheck
