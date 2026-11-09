// src/core/risk_concentration.cpp - see include/strata/core/risk_concentration.hpp.
#include "strata/core/risk_concentration.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>

namespace strata::core::risk {

RowShares row_shares(const std::vector<double>& x) {
    RowShares r;
    r.events = (int) x.size();
    std::vector<double> v;
    v.reserve(x.size());
    double total = 0.0;
    for (double a : x) {
        const double c = a > 0.0 ? a : 0.0;
        v.push_back(c);
        total += c;
    }
    if (total <= 0.0) return r;
    std::sort(v.begin(), v.end(), std::greater<double>());
    double run = 0.0;
    bool got50 = false, got80 = false, got90 = false;
    for (size_t i = 0; i < v.size(); ++i) {
        run += v[i];
        const int k = (int) i + 1;
        if (k == 1) r.top1 = run / total;
        if (k == 3) r.top3 = run / total;
        if (k == 10) r.top10 = run / total;
        if (!got50 && run >= 0.5 * total) { r.need50 = k; got50 = true; }
        if (!got80 && run >= 0.8 * total) { r.need80 = k; got80 = true; }
        if (!got90 && run >= 0.9 * total) { r.need90 = k; got90 = true; }
    }
    // fewer than 3 or 10 events: the top-k carry all the risk
    if (v.size() < 3) r.top3 = 1.0;
    if (v.size() < 10) r.top10 = 1.0;
    // (a row's last element rounding can leave run a hair below 0.9 * total: the count is then all of them)
    if (!got50) r.need50 = (int) v.size();
    if (!got80) r.need80 = (int) v.size();
    if (!got90) r.need90 = (int) v.size();
    return r;
}

namespace {
int decile(double s) {
    int b = (int) (s * 10.0);
    return b < 0 ? 0 : (b > 9 ? 9 : b);
}
}  // namespace

void Concentration::add(const RowShares& r, bool flipped_row) {
    Group& g = flipped_row ? flipped : held;
    ++g.rows;
    g.sum_top1 += r.top1;
    g.sum_top3 += r.top3;
    g.sum_top10 += r.top10;
    g.sum_events += r.events;
    g.sum_need50 += r.need50;
    g.sum_need80 += r.need80;
    g.sum_need90 += r.need90;
    ++g.hist_top1[decile(r.top1)];
    ++g.hist_top3[decile(r.top3)];
    ++g.hist_top10[decile(r.top10)];
}

namespace {
std::string group_part(const char* name, const Group& g) {
    if (g.rows == 0) return "";
    const double n = (double) g.rows;
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s %lld rows: top1 %.2f, top3 %.2f, top10 %.2f, events to 50/80/90%% %.1f/%.1f/%.1f of %.1f",
                  name, (long long) g.rows, g.sum_top1 / n, g.sum_top3 / n, g.sum_top10 / n, g.sum_need50 / n, g.sum_need80 / n,
                  g.sum_need90 / n, g.sum_events / n);
    return buf;
}
}  // namespace

std::string Concentration::line() const {
    const int64_t rows = flipped.rows + held.rows;
    if (rows == 0) return "";
    const double n = (double) rows;
    const double t1 = (flipped.sum_top1 + held.sum_top1) / n, t3 = (flipped.sum_top3 + held.sum_top3) / n,
                 t10 = (flipped.sum_top10 + held.sum_top10) / n;
    const double ev = (flipped.sum_events + held.sum_events) / n;
    char head[256];
    std::snprintf(head, sizeof head,
                  "swap risk concentration (cut rows %lld): top1 share mean %.2f, top3 %.2f, top10 %.2f; events to carry 50/80/90%% of the risk: "
                  "%.1f/%.1f/%.1f of %.1f per row",
                  (long long) rows, t1, t3, t10, (flipped.sum_need50 + held.sum_need50) / n, (flipped.sum_need80 + held.sum_need80) / n,
                  (flipped.sum_need90 + held.sum_need90) / n, ev);
    std::string out = head;
    const std::string f = group_part("flipped", flipped), h = group_part("held", held);
    if (!f.empty()) out += "; " + f;
    if (!h.empty()) out += "; " + h;
    return out;
}

}  // namespace strata::core::risk
