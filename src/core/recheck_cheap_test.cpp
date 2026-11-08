// recheck_cheap_test - SPEC-recheck-cheap on the CPU (include/strata/core/recheck_cheap.hpp): the exact successor's plan (row 0 only by
// default; the whole-window exact path with STRATA_RECHECK_ROW_ONLY=0), the control row count (every row), and the prefetch list of a cut
// row (dedup, layer order, the cap, the resident skip). No CUDA, no engine.
#include "strata/core/recheck_cheap.hpp"

#include <cstdio>
#include <cstdlib>

namespace rc = strata::core::recheck;

namespace {
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "recheck_cheap_test: FAILED: %s\n", what);
        std::exit(1);
    }
}
}  // namespace

int main() {
    // the successor: row-only keeps the budget (no zero budget, no whole-window exact); the old path is the zero budget and win_exact
    {
        const rc::SuccessorPlan a = rc::successor_plan(true, true);
        check(a.row_only && !a.win_exact && !a.zero_budget, "successor, row-only: row 0 only, no whole-window exact, budget kept");
        const rc::SuccessorPlan b = rc::successor_plan(true, false);
        check(!b.row_only && b.win_exact && b.zero_budget, "successor, STRATA_RECHECK_ROW_ONLY=0 (or no flag): the whole-window exact path of before");
        const rc::SuccessorPlan c = rc::successor_plan(false, true);
        check(!c.row_only && !c.win_exact && !c.zero_budget, "a window that is not a successor is untouched, row-only or not");
        const rc::SuccessorPlan d = rc::successor_plan(false, false);
        check(!d.row_only && !d.win_exact && !d.zero_budget, "a window that is not a successor is untouched, old switch");
    }

    // the control window: every row (the flag's count is at least any window's rows)
    check(rc::kAllRows > 8 * 4, "the control row count covers every row of any window");

    // the prefetch list: picks (layer, expert) in the order the cut row's event list holds them
    {
        const std::vector<rc::Pick> picks = {{2, 5}, {1, 7}, {2, 5}, {1, 9}, {3, 1}, {1, 7}};
        auto none = [](int32_t, int32_t) { return false; };
        const rc::PrefetchPlan p = rc::plan_prefetch(picks, 96, none);
        check(p.todo.size() == 4 && p.duplicate == 2 && p.resident == 0 && p.over_cap == 0, "prefetch: 4 fresh picks, 2 repeats");
        check(p.todo[0].layer == 1 && p.todo[0].expert == 7 && p.todo[1].layer == 1 && p.todo[1].expert == 9 &&
                  p.todo[2].layer == 2 && p.todo[2].expert == 5 && p.todo[3].layer == 3 && p.todo[3].expert == 1,
              "prefetch: ascending layer order, first appearance within a layer");
    }
    {
        const std::vector<rc::Pick> picks = {{2, 5}, {1, 7}, {2, 5}, {1, 9}, {3, 1}, {1, 7}};
        auto resident19 = [](int32_t l, int32_t e) { return l == 1 && e == 9; };
        const rc::PrefetchPlan p = rc::plan_prefetch(picks, 96, resident19);
        check(p.todo.size() == 3 && p.resident == 1 && p.duplicate == 2, "prefetch: a resident pick is skipped and counted, not queued");
        for (const rc::Pick& k : p.todo) check(!(k.layer == 1 && k.expert == 9), "prefetch: the resident pick is not in the list");
    }
    {
        const std::vector<rc::Pick> picks = {{2, 5}, {1, 7}, {3, 1}, {4, 4}};
        auto none = [](int32_t, int32_t) { return false; };
        const rc::PrefetchPlan p = rc::plan_prefetch(picks, 2, none);
        check(p.todo.size() == 2 && p.over_cap == 2, "prefetch: the cap keeps the first two in layer order, the rest are dropped and counted");
        check(p.todo[0].layer == 1 && p.todo[1].layer == 2, "prefetch: the cap takes the lowest layers");
        const rc::PrefetchPlan z = rc::plan_prefetch(picks, 0, none);
        check(z.todo.empty() && z.over_cap == 4, "prefetch: cap 0 queues nothing");
        const rc::PrefetchPlan e = rc::plan_prefetch({}, 96, none);
        check(e.todo.empty() && e.resident == 0 && e.duplicate == 0 && e.over_cap == 0, "prefetch: an empty row queues nothing");
    }

    std::printf("recheck_cheap_test: ok\n");
    return 0;
}
