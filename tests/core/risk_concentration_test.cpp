// tests/core/risk_concentration_test.cpp - the per-row share of swap risk carried by the top events (CPU only).
#include "strata/core/risk_concentration.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using strata::core::risk::Concentration;
using strata::core::risk::row_shares;
using strata::core::risk::RowShares;

namespace {
int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

bool near(double a, double b) { return std::fabs(a - b) < 1e-12; }

void test_shares() {
    RowShares e = row_shares({});
    CHECK(e.events == 0 && near(e.top1, 0) && e.need50 == 0, "no events: all zero");

    RowShares z = row_shares({0.0, 0.0});
    CHECK(z.events == 2 && near(z.top1, 0) && z.need90 == 0, "no risk at all: shares 0, counts 0");

    // total 10: sorted 4,2,1,1,1,1
    RowShares r = row_shares({1, 4, 1, 2, 1, 1});
    CHECK(near(r.top1, 0.4), "top1 = 4/10");
    CHECK(near(r.top3, 0.7), "top3 = (4+2+1)/10");
    CHECK(near(r.top10, 1.0), "fewer than 10 events: top10 is everything");
    CHECK(r.need50 == 2, "50%: cumulative 4, 6 -> 2 events");
    CHECK(r.need80 == 4, "80%: cumulative 4, 6, 7, 8 -> 4 events");
    CHECK(r.need90 == 5, "90%: cumulative reaches 9 at 5 events");
    CHECK(r.events == 6, "six events");

    // a negative risk counts as zero
    RowShares n = row_shares({-5, 2, 2});
    CHECK(near(n.top1, 0.5) && near(n.top3, 1.0), "negative risk is 0 (shares of the rest)");

    // ten events of equal risk: top10 = 1, top1 = 0.1
    std::vector<double> eq(10, 1.0);
    RowShares q = row_shares(eq);
    CHECK(near(q.top1, 0.1) && near(q.top3, 0.3) && near(q.top10, 1.0), "equal risks spread evenly");
    CHECK(q.need50 == 5 && q.need90 == 9, "equal risks: 5 for 50%, 9 for 90%");

    // a single event carries everything
    RowShares s = row_shares({3.0});
    CHECK(near(s.top1, 1.0) && s.need50 == 1 && s.need90 == 1, "one event: share 1, need 1");
}

void test_groups_and_line() {
    Concentration c;
    CHECK(c.line().empty(), "no rows: no line");
    c.add(row_shares({4, 2, 1, 1, 1, 1}), true);   // flipped: top1 0.4
    c.add(row_shares({1, 1}), false);              // held: top1 0.5
    CHECK(c.flipped.rows == 1 && c.held.rows == 1, "rows split by the verdict");
    // share 0.4 -> decile 4; 0.5 -> decile 5; top10 share 1.0 -> decile 9
    CHECK(c.flipped.hist_top1[4] == 1 && c.held.hist_top1[5] == 1, "top1 deciles");
    CHECK(c.flipped.hist_top10[9] == 1 && c.held.hist_top10[9] == 1, "a share of 1.0 falls in the last decile");
    const std::string line = c.line();
    CHECK(line.find("swap risk concentration (cut rows 2)") != std::string::npos, "line head with the row count");
    CHECK(line.find("events to carry 50/80/90%") != std::string::npos, "line names the 50/80/90 events");
    CHECK(line.find("flipped 1 rows") != std::string::npos && line.find("held 1 rows") != std::string::npos, "line splits the groups");
    c.reset();
    CHECK(c.line().empty(), "reset clears the request");
}

}  // namespace

int main() {
    test_shares();
    test_groups_and_line();
    if (failures == 0) {
        std::printf("risk_concentration_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "risk_concentration_test: %d check(s) failed\n", failures);
    return 1;
}
