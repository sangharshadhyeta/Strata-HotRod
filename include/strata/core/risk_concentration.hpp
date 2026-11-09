// include/strata/core/risk_concentration.hpp - how concentrated a cut row's swap risk is (coordinator's counter, SPEC-substitute-fit stage 1a).
//
// At each cut the row's events carry risk x (w x gap, the mask-learn MaskEv list).  Per row this gives the share of the total risk the
// top 1, top 3 and top 10 events carry, and how many events carry 50%, 80% and 90% of it.  Rows are kept apart by whether the exact
// successor flipped (the verdict arrives at the successor's settle).  Histograms of the shares (deciles) are accumulated per group.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core::risk {

/// One cut row's concentration.  With no risk at all (total 0) the shares are 0 and the counts 0.
struct RowShares {
    int events = 0;                   ///< m: the row's events (all of them, risk 0 included)
    double top1 = 0, top3 = 0, top10 = 0;
    int need50 = 0, need80 = 0, need90 = 0;   ///< the fewest events (largest first) that carry that share of the total
};

/// x: the events' risks (any order, >= 0 expected; negatives count as 0).
RowShares row_shares(const std::vector<double>& x);

/// The request's accumulator: per group (flipped, not flipped) the row count, the sums, and the decile histograms of the three shares.
struct Group {
    int64_t rows = 0;
    double sum_top1 = 0, sum_top3 = 0, sum_top10 = 0;
    double sum_events = 0, sum_need50 = 0, sum_need80 = 0, sum_need90 = 0;
    int64_t hist_top1[10] = {}, hist_top3[10] = {}, hist_top10[10] = {};   ///< decile bins: share 1.0 falls in the last bin
};

struct Concentration {
    Group flipped, held;   ///< flipped: the exact successor's top-1 differed from the cut row's; held: it did not
    /// Adds one row to its group.
    void add(const RowShares& r, bool flipped_row);
    /// The request-end line (no trailing newline).  Empty when no row was counted.
    std::string line() const;
    /// Clears the counts (the request's end).
    void reset() { *this = Concentration{}; }
};

}  // namespace strata::core::risk
