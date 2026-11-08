// recheck_gate_test - SPEC-recheck-gated on the CPU (include/strata/core/recheck_gate.hpp): the per-row rule from the drafter probability,
// the hold bitmask, eligibility, the audit cadence, the Wilson bounds, the threshold mover (rule 5: scripted counts, bounds, range, one move
// per 2,000 audited moved rows), the accumulation across requests, the early-margin deciles, and the save / load round trip.
// No CUDA, no engine.
#include "strata/core/recheck_gate.hpp"
#include "strata/kernels/native_router.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace g = strata::core::gate;

namespace {
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "recheck_gate_test: FAILED: %s\n", what);
        std::exit(1);
    }
}
bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
}  // namespace

int main() {
    // rule 1-3: confident (p >= P_HI), uncertain (p < P_HI), none (no drafter probability, whatever p says)
    check(g::classify(9, 0.95f, 0.9f) == g::Rule::Confident, "rule 1: p 0.95 >= P_HI 0.9 is confident");
    check(g::classify(9, 0.9f, 0.9f) == g::Rule::Confident, "rule 1: p exactly P_HI is confident");
    check(g::classify(8, 0.85f, 0.9f) == g::Rule::Uncertain, "rule 2: p 0.85 < P_HI is uncertain");
    check(g::classify(g::kNone, 0.0f, 0.9f) == g::Rule::None, "rule 3: no drafter probability is none, whatever the value");
    check(g::classify(0, 0.05f, 0.6f) == g::Rule::Uncertain, "rule 2: the lowest bucket is uncertain at P_HI 0.6");

    // eligibility (8 Nov rule): confident rows only in an audit window; uncertain rows and none rows always
    check(!g::eligible(g::Rule::Confident, false), "rule 1 rows are not eligible outside an audit window");
    check(g::eligible(g::Rule::Confident, true), "rule 1 rows are eligible in an audit window");
    check(g::eligible(g::Rule::Uncertain, false) && g::eligible(g::Rule::Uncertain, true),
          "rule 2 rows are eligible for a cut, as today (not held)");
    check(g::eligible(g::Rule::None, false) && g::eligible(g::Rule::None, true), "rule 3 rows are always eligible");

    // rule 2 as the hold mask: no row is held any more (uncertain rows stay eligible); the function returns 0 whatever the rows
    {
        const int bucket[5] = {9, 3, g::kNone, 8, 2};
        const float p[5] = {0.97f, 0.35f, 0.0f, 0.88f, 0.22f};
        check(g::uncertain_bits(bucket, p, 5, 0.9f, false) == 0u, "hold mask: no uncertain row is held (rule 2 changed)");
        check(g::uncertain_bits(bucket, p, 5, 0.9f, true) == 0u, "hold mask: an audit window holds nothing");
        check(g::uncertain_bits(bucket, p, 5, 0.975f, false) == 0u, "hold mask: nothing held at any P_HI");
    }

    // rule 4: the audit cadence
    check(!g::audit_due(255, 256) && g::audit_due(256, 256) && !g::audit_due(257, 256) && g::audit_due(512, 256),
          "audit cadence: every 256th decode window (256, 512, ...)");
    check(!g::audit_due(256, 0) && !g::audit_due(0, 256), "audit cadence: off with every 0, and never on window 0");

    // the range and the buckets of P_HI
    check(near(g::clamp_p_hi(0.5f), 0.6, 1e-6) && near(g::clamp_p_hi(0.99f), 0.95, 1e-6) && near(g::clamp_p_hi(0.8f), 0.8, 1e-6),
          "range: P_HI clamped to 0.6-0.95");
    check(g::below_bucket(0.9f) == 8 && g::confident_bucket(0.9f) == 9 && g::below_bucket(0.8f) == 7 && g::confident_bucket(0.6f) == 6,
          "buckets: the bucket just below P_HI and the first confident bucket");

    // Wilson 90% (z = 1.645): zero flips in n rows bounds the rate at about 2.7 / (n + 2.7); 0 of 540 is under 0.5%, 0 of 500 is not
    check(g::wilson(0, 540, true) <= g::kFlipBound && g::wilson(0, 500, true) > g::kFlipBound, "Wilson upper: 0 flips in 540 rows is under 0.5%, in 500 not");
    check(near(g::wilson(0, 1000, true), 2.706 / 1002.706, 1e-3), "Wilson upper: 0 of 1000 is about 0.27%");
    check(g::wilson(0, 1000, false) == 0.0 || g::wilson(0, 1000, false) < 1e-9, "Wilson lower: 0 flips gives 0");
    check(near(g::wilson(10, 1000, false), 0.0064, 0.0015), "Wilson lower: 10 of 1000 is about 0.6%");
    check(g::wilson(0, 0, true) == 1.0 && g::wilson(0, 0, false) == 0.0, "Wilson: no rows, the widest bounds");

    // rule 5, the mover: needs 2,000 moved rows in the drafter buckets since the last move
    {
        // a move down: P_HI 0.9, bucket 8 (just below) with 1,000 moved rows and no flip; the confident bucket 9 is clean too
        g::State s;
        s.since.moved[8] = 1000;
        s.since.moved[9] = 1000;
        check(g::step(s) == -1 && near(s.p_hi, 0.8, 1e-6) && s.moves == 1, "mover: down to 0.8 on 0 flips in 1,000 rows just below");
        check(s.since.moved[8] == 0 && s.since.moved[9] == 0, "mover: a move resets the evidence since the last move");
        check(g::step(s) == 0, "mover: no evidence after a move, no second move");
    }
    {
        // the 2,000 rule: 1,500 moved rows with clean evidence do not move; the same evidence at 2,000 does
        g::State s;
        s.since.moved[8] = 1500;
        check(g::step(s) == 0 && s.moves == 0, "mover: fewer than 2,000 moved rows, no move");
        s.since.moved[8] = 2000;
        check(g::step(s) == -1, "mover: 2,000 moved rows, the same clean evidence moves");
    }
    {
        // a move down is refused when the bucket below has too many flips: 6 flips in 1,000 is a 0.6% rate, upper bound above 0.5%
        g::State s;
        s.since.moved[8] = 1000;
        s.since.flip[8] = 6;
        check(g::step(s) == 0, "mover: 6 flips in 1,000 just below blocks the move down");
    }
    {
        // a move up: the confident bucket 9 flips 20 times in 1,000 moved rows, the lower bound is above 0.5%
        g::State s;
        s.since.moved[9] = 1000;
        s.since.flip[9] = 20;
        s.since.moved[8] = 1000;
        check(g::step(s) == 1 && near(s.p_hi, 0.95, 1e-6), "mover: confident flips above the bound move P_HI up to 0.95");
        check(g::step(s) == 0, "mover: at the 0.95 cap, no further move up");
    }
    {
        // the range floor: at 0.6 there is no move down
        g::State s;
        s.p_hi = 0.6f;
        s.since.moved[5] = 3000;
        check(g::step(s) == 0 && near(s.p_hi, 0.6, 1e-6), "mover: the floor 0.6 holds");
    }
    {
        // the up test has priority: confident flips and a clean bucket below cannot move down at once
        g::State s;
        s.since.moved[8] = 1000;
        s.since.moved[9] = 1000;
        s.since.flip[9] = 30;
        check(g::step(s) == 1, "mover: a bad confident bucket moves up before any move down");
    }

    // accumulation across requests: the totals keep every eligible row; `since` restarts at each move
    {
        g::State s;
        for (int i = 0; i < 3; ++i) g::add_moved(s, 7);
        g::add_cut(s, 7);
        g::add_flip(s, 7);
        g::add_moved(s, g::kNone);
        check(s.total.moved[7] == 3 && s.total.cut[7] == 1 && s.total.flip[7] == 1 && s.total.moved[g::kNone] == 1,
              "accumulation: the totals count the eligible rows per bucket");
        check(s.since.moved[7] == 3, "accumulation: since counts the same rows until a move");
        s.since.moved[8] = 2500;   // a move resets since, not the totals
        g::step(s);
        check(s.total.moved[7] == 3 && s.since.moved[7] == 0, "accumulation: a move keeps the totals and clears since");
    }

    // early-margin deciles: the share of the ring below the margin, tenths; NaN is not kept
    {
        g::State s;
        check(g::margin_decile(s, 1.0f) == 0, "deciles: no sample gives decile 0");
        for (int i = 0; i < 100; ++i) g::add_margin_sample(s, static_cast<float>(i));
        g::add_margin_sample(s, 0.0f / 0.0f);
        check(s.margin_ring.size() == 100, "deciles: NaN margins are not kept");
        check(g::margin_decile(s, 0.5f) == 0 && g::margin_decile(s, 25.5f) == 2 && g::margin_decile(s, 95.5f) == 9 &&
                  g::margin_decile(s, 1000.0f) == 9,
              "deciles: the share of the ring below the margin, in tenths");
        for (int i = 0; i < g::kMarginRing + 10; ++i) g::add_margin_sample(s, 1.0f);
        check(s.margin_ring.size() == (size_t) g::kMarginRing, "deciles: the ring keeps at most kMarginRing samples");
    }

    // persistence: a save and a load round trip is exact; another model's fingerprint is refused; a malformed file is refused
    {
        g::State s;
        s.p_hi = 0.8f;
        s.moves = 3;
        s.total.moved[2] = 11;
        s.total.cut[2] = 4;
        s.total.flip[2] = 1;
        s.since.moved[8] = 900;
        s.since.flip[8] = 2;
        s.margin.moved[4] = 7;
        s.margin.cut[4] = 2;
        s.margin.flip[4] = 1;
        for (int i = 0; i < 37; ++i) g::add_margin_sample(s, 0.1f * (float) i);
        s.margin_pos = 5;
        const std::string text = g::serialize(s, 0xabcdefULL);
        g::State t;
        check(g::parse(text, 0xabcdefULL, t), "round trip: the file parses for its own fingerprint");
        check(near(t.p_hi, 0.8, 1e-6) && t.moves == 3 && t.total.moved[2] == 11 && t.total.cut[2] == 4 && t.total.flip[2] == 1 &&
                  t.since.moved[8] == 900 && t.since.flip[8] == 2 && t.margin.moved[4] == 7 && t.margin.cut[4] == 2 && t.margin.flip[4] == 1 &&
                  t.margin_pos == 5,
              "round trip: counts, P_HI, moves and the margin position come back");
        check(t.margin_ring.size() == s.margin_ring.size(), "round trip: the margin ring keeps its length");
        bool same = true;
        for (size_t i = 0; i < s.margin_ring.size(); ++i) same = same && t.margin_ring[i] == s.margin_ring[i];
        check(same, "round trip: the margin ring keeps its values exactly");
        check(g::serialize(t, 0xabcdefULL) == text, "round trip: serialize of the loaded state is the same text");

        g::State other;
        other.p_hi = 0.5f;
        check(!g::parse(text, 0x123ULL, other) && near(other.p_hi, 0.5, 1e-6), "fingerprint: another model's file is refused, state unchanged");
        check(!g::parse("p_hi 0.9\n", 0xabcdefULL, other), "fingerprint: a file with no fingerprint line is refused");
        check(!g::parse("fingerprint abcdef\ntotal_moved 1 2\n", 0xabcdefULL, other), "round trip: a short counts line is refused");
    }

    // SPEC-recheck-gated (D): the rank packed into MaskEv::kind (bit 0 keeps the kind, bits 1-4 the rank) round-trips for every rank and kind
    for (int kind = 0; kind <= 1; ++kind)
        for (int rank = 0; rank <= 9; ++rank) {
            const int32_t v = strata::kernels::mask_ev_pack(kind, rank);
            check(strata::kernels::mask_ev_kind(v) == kind && strata::kernels::mask_ev_rank(v) == rank, "rank packing: kind and rank round trip");
        }
    check(strata::kernels::mask_ev_pack(1, 0) == 1 && strata::kernels::mask_ev_pack(0, 9) == 18, "rank packing: kind in bit 0, rank from bit 1");
    // the rank slots (3-9) and the layer bands (0-11, 12-23, 24-35, 36-47)
    check(g::rank_slot(3) == 0 && g::rank_slot(9) == 6 && g::rank_slot(2) == -1 && g::rank_slot(0) == -1, "rank slots: ranks 3-9 only");
    check(g::band_of(0) == 0 && g::band_of(11) == 0 && g::band_of(12) == 1 && g::band_of(47) == 3 && g::band_of(60) == 3,
          "layer bands: 12 layers each, clamped to the last band");
    {
        // the rank table survives a save and load (its own line)
        g::State s;
        s.rank.n[4][2][1] = 5;
        s.rank.n[0][0][0] = 2;
        g::State t;
        check(g::parse(g::serialize(s, 7), 7, t) && t.rank.n[4][2][1] == 5 && t.rank.n[0][0][0] == 2 && t.rank.n[6][3][1] == 0,
              "round trip: the rank table comes back");
    }

    // SPEC-recheck-gated step F: the shift ring (median, deciles, the cap of the newest samples), the caps per class, the window's caps
    {
        g::Ring r;
        for (int i = 1; i <= 4; ++i) r.add((float) i, 4);
        r.add(5.0f, 4);
        r.add(6.0f, 4);
        check(r.n() == 4 && r.pos == 2, "shift ring: the newest 4 kept, the next slot after the wrap");
        check(r.median() == 5.0f, "shift ring: the upper median of {3,4,5,6} is 5");
        g::Ring z;
        z.add(0.0f, 8);
        z.add(-1.0f, 8);
        z.add(0.0f / 0.0f, 8);
        check(z.n() == 0 && z.median() == 0.0f && z.decile(1.0f) == 0, "shift ring: zero, negative and NaN shifts are not kept");
        g::Ring d;
        for (int i = 1; i <= 10; ++i) d.add((float) i, 64);
        check(d.decile(5.5f) == 5 && d.decile(0.5f) == 0 && d.decile(100.0f) == 9, "shift ring: deciles from the share below");
    }
    {
        const g::CapMult m;   // defaults: conf +inf, unc 1.5, none 1.0
        check(g::cap_for(g::Rule::Uncertain, 2.0f, 10, m) == INFINITY, "cap: no cap before 64 samples");
        check(g::cap_for(g::Rule::Uncertain, 0.0f, 100, m) == INFINITY, "cap: no cap with a zero median");
        check(g::cap_for(g::Rule::Confident, 2.0f, 100, m) == INFINITY, "cap: confident rows default to no cap");
        check(near(g::cap_for(g::Rule::Uncertain, 2.0f, 100, m), 3.0, 1e-6), "cap: uncertain rows at 1.5 x the median");
        check(near(g::cap_for(g::Rule::None, 2.0f, 100, m), 2.0, 1e-6), "cap: none rows at 1.0 x the median");
        g::CapMult mc;
        mc.conf = 2.0f;
        check(near(g::cap_for(g::Rule::Confident, 2.0f, 100, mc), 4.0, 1e-6), "cap: a finite confident multiplier applies");
    }
    {
        const int bucket[3] = {9, 3, g::kNone};
        const float p[3] = {0.95f, 0.3f, 0.0f};
        float caps[32];
        g::fill_caps(caps, bucket, p, 3, 0.9f, false, 2.0f, 100, g::CapMult{});
        check(caps[0] == INFINITY && near(caps[1], 3.0, 1e-6) && near(caps[2], 2.0, 1e-6) && caps[3] == INFINITY && caps[31] == INFINITY,
              "fill_caps: confident inf, uncertain 1.5 x, none 1 x; rows past T inf");
        g::fill_caps(caps, bucket, p, 3, 0.9f, true, 2.0f, 100, g::CapMult{});
        bool all_inf = true;
        for (float v : caps) all_inf = all_inf && v == INFINITY;
        check(all_inf, "fill_caps: an audit window caps nothing");
        g::fill_caps(caps, bucket, p, 3, 0.9f, false, 2.0f, 10, g::CapMult{});
        check(caps[1] == INFINITY, "fill_caps: no cap until the median has 64 samples");
    }
    {
        // the calibration counter and the shift ring survive a save and a load
        g::State s;
        s.cal.flip[7] = 3;
        s.cal.not_flip[7] = 40;
        for (int i = 1; i <= 70; ++i) s.shift.add(0.01f * (float) i, g::kShiftRing);
        g::State t;
        check(g::parse(g::serialize(s, 9), 9, t) && t.cal.flip[7] == 3 && t.cal.not_flip[7] == 40 && t.shift.n() == 70 &&
                  t.shift.median() == s.shift.median() && t.shift.pos == s.shift.pos,
              "round trip: calibration and the shift ring come back");
    }

    // SPEC-recheck-gated step G: the rotation budget -> mask -> cap -> prefetch -> budget, skipping the kinds that are off
    {
        namespace t = strata::core::tune;
        check(t::next_kind(t::Budget, true, true, true, true) == t::Mask && t::next_kind(t::Mask, true, true, true, true) == t::Cap &&
                  t::next_kind(t::Cap, true, true, true, true) == t::Prefetch && t::next_kind(t::Prefetch, true, true, true, true) == t::KMult &&
                  t::next_kind(t::KMult, true, true, true, true) == t::Budget,
              "rotation: budget -> mask -> cap -> prefetch -> k -> budget (step H2: the fifth kind)");
        check(t::next_kind(t::Budget, false, true, true, true) == t::Cap && t::next_kind(t::Cap, false, true, true, true) == t::Prefetch,
              "rotation: mask off is skipped");
        check(t::next_kind(t::Mask, true, false, true, true) == t::Prefetch && t::next_kind(t::Prefetch, true, false, false, false) == t::Budget,
              "rotation: cap off is skipped; prefetch off goes back to budget");
        check(t::next_kind(t::Prefetch, false, false, false, true) == t::KMult && t::next_kind(t::KMult, false, false, false, false) == t::Budget,
              "rotation: k alone after prefetch; k off goes back to budget");
        check(t::next_kind(t::Budget, false, false, false) == t::Budget && t::next_kind(t::Cap, false, false, false) == t::Budget,
              "rotation: only budget on, always budget");
    }
    {
        namespace t = strata::core::tune;
        // the cap candidate: both multipliers times step^d, the ratio kept (1.5), up 1.25 / down 1 / 1.25
        double cu = 0.0, cn = 0.0;
        check(t::cap_candidate(1.5, 1.0, 1.25, 1, cu, cn) && near(cu, 1.875, 1e-9) && near(cn, 1.25, 1e-9) && near(cu / cn, 1.5, 1e-9),
              "cap candidate: up by the step, ratio 1.5 kept");
        check(t::cap_candidate(1.5, 1.0, 1.25, -1, cu, cn) && near(cu, 1.2, 1e-9) && near(cn, 0.8, 1e-9), "cap candidate: down by the step");
        check(t::cap_candidate(7.0, 1.0, 1.25, 1, cu, cn) && cu <= 8.0 + 1e-9 && near(cu / cn, 7.0, 1e-9),
              "cap candidate: the larger multiplier stops at 8, the ratio kept");
        check(!t::cap_candidate(0.3, 0.25, 1.25, -1, cu, cn), "cap candidate: at the 0.25 floor a down move is not a candidate");
        check(!t::cap_candidate(0.0, 1.0, 1.25, 1, cu, cn) && !t::cap_candidate(1.5, 1.0, 0.0, 1, cu, cn),
              "cap candidate: a non-positive multiplier or step is not a candidate");
        // the prefetch candidate: x or / 1.5 rounded, at least one step, range 0-64, 0 a candidate
        long long m = -1;
        check(t::prefetch_candidate(12, 1, m) && m == 18, "prefetch candidate: 12 up 1.5x is 18");
        check(t::prefetch_candidate(12, -1, m) && m == 8, "prefetch candidate: 12 down 1.5x is 8");
        check(t::prefetch_candidate(2, 1, m) && m == 3, "prefetch candidate: at least one step (2 up is 3)");
        check(t::prefetch_candidate(1, -1, m) && m == 0, "prefetch candidate: 0 is a candidate (1 down is 0)");
        check(!t::prefetch_candidate(0, -1, m), "prefetch candidate: 0 down is not a candidate");
        check(!t::prefetch_candidate(64, 1, m) && t::prefetch_candidate(40, 1, m) && m == 60, "prefetch candidate: range 0-64");
    }
    {
        // the tune state survives a save and a load; an out-of-range prefetch count is refused
        g::State s;
        s.tune.set = true;
        s.tune.cap_unc = 1.875;
        s.tune.cap_none = 1.25;
        s.tune.cap_step = 1.5625f;
        s.tune.cap_last_win = 1;
        s.tune.cap_first = -1;
        s.tune.cap_stage = 1;
        s.tune.prefetch = 18;
        s.tune.pf_first = -1;
        s.tune.pf_stage = 1;
        g::State u;
        check(g::parse(g::serialize(s, 11), 11, u) && u.tune.set && near(u.tune.cap_unc, 1.875, 1e-6) && near(u.tune.cap_none, 1.25, 1e-6) &&
                  near(u.tune.cap_step, 1.5625, 1e-6) && u.tune.cap_last_win == 1 && u.tune.cap_first == -1 && u.tune.cap_stage == 1 &&
                  u.tune.prefetch == 18 && u.tune.pf_first == -1 && u.tune.pf_stage == 1,
              "round trip: the tune state (cap multipliers, steps, directions, prefetch count) comes back");
        g::State v;
        check(!g::parse("fingerprint 0b\ntune_prefetch 99 1 0\n", 11, v), "tune: a prefetch count above 64 is refused");
        check(!g::parse("fingerprint 0b\ntune_cap 0 1 1.25 0 1 0\n", 11, v), "tune: a non-positive cap multiplier is refused");
    }

    // SPEC-recheck-gated step H1: k from evidence above the noise floor (k_evidence, k_target)
    {
        // flips concentrated below ratio 0.02: all 40 entries at 0.02 flipped -> r* 0.02, k 1.25 x 0.02 = 0.025
        std::vector<float> ratio(40, 0.02f);
        std::vector<unsigned char> flip(40, true);
        const g::KEvidence e = g::k_evidence(ratio.data(), reinterpret_cast<const bool*>(flip.data()), 40, 0.074);
        check(e.found && near(e.r_star, 0.02, 1e-6) && e.n_t == 40 && e.f_t == 40, "k_evidence: flips at 0.02 give r* 0.02");
        check(near(g::k_target(e, 40, 1.0f, 30), 0.025, 1e-6), "k_target: 1.25 x r* = 0.025");
    }
    {
        // noise at high ratios: 200 entries with ratio 0.005 .. 1.0; from ratio 0.505 on, 1 in 14 flipped (7%, the control's rate); no flip below.
        // No threshold's entries show a flip rate above the control's noise: no evidence, so k halves from a full ring.
        std::vector<float> ratio(200);
        std::vector<unsigned char> flip(200, false);
        for (int i = 0; i < 200; ++i) {
            ratio[(size_t) i] = 0.005f * (float) (i + 1);
            flip[(size_t) i] = i >= 100 && i % 14 == 0;
        }
        const g::KEvidence e = g::k_evidence(ratio.data(), reinterpret_cast<const bool*>(flip.data()), 200, 0.074);
        check(!e.found, "k_evidence: flips at the control's noise rate, at high ratios, give no evidence");
        check(near(g::k_target(e, 200, 1.0f, 30), 0.5, 1e-6), "k_target: no evidence with a full ring and measured noise halves k");
        check(near(g::k_target(e, 200, 1.0f, 19), 1.0, 1e-6), "k_target: no evidence but fewer than 20 control samples holds k (8 Nov fix)");
        check(near(g::k_target(e, 200, 1.0f, 0), 1.0, 1e-6), "k_target: no control samples holds k");
        check(near(g::k_target(e, 200, 0.007f, 30), 0.005, 1e-6), "k_target: halving stops at the 0.005 floor");
        check(near(g::k_target(e, 200, 0.005f, 30), 0.005, 1e-6), "k_target: k at the floor stays");
        check(near(g::k_target(e, 200, 0.0012f, 30), 0.0012, 1e-6), "k_target: k already below the floor is not lowered further");
    }
    {
        // small n: 5 entries, all flipped: fewer than 8 at or above any threshold, so no evidence; a short ring keeps k
        std::vector<float> ratio(5, 0.3f);
        std::vector<unsigned char> flip(5, true);
        const g::KEvidence e = g::k_evidence(ratio.data(), reinterpret_cast<const bool*>(flip.data()), 5, 0.0);
        check(!e.found && near(g::k_target(e, 5, 0.7f, 30), 0.7, 1e-6), "k_evidence: fewer than 8 entries give no evidence, k kept");
        // no control samples (ctrl_hi 1): nothing qualifies, as before (no raise)
        std::vector<float> r40(40, 0.02f);
        std::vector<unsigned char> f40(40, true);
        check(!g::k_evidence(r40.data(), reinterpret_cast<const bool*>(f40.data()), 40, 1.0).found, "k_evidence: no control samples, no raise");
        // no flips at all in a full ring: no evidence, k halves
        std::vector<float> rn(60, 0.1f);
        std::vector<unsigned char> fn(60, false);
        check(!g::k_evidence(rn.data(), reinterpret_cast<const bool*>(fn.data()), 60, 0.074).found, "k_evidence: no flip, no evidence");
        // the clamp: 1.25 x r* above 50 is 50
        check(near(g::k_target(g::KEvidence{true, 100.0, 9, 9}, 9, 1.0f, 30), 50.0, 1e-6), "k_target: clamped to 50");
    }
    // SPEC-recheck-gated step H2: the margin m candidates (m x step^d, range 0.6-1.5), the explore veto, and the persistence of m
    {
        namespace t = strata::core::tune;
        double m = 0.0;
        check(t::k_mult_candidate(1.0, 1.25, 1, m) && near(m, 1.25, 1e-9), "k candidate: up by the step");
        check(t::k_mult_candidate(1.0, 1.25, -1, m) && near(m, 0.8, 1e-9), "k candidate: down by the step");
        check(!t::k_mult_candidate(1.5, 1.25, 1, m) && !t::k_mult_candidate(0.6, 1.25, -1, m), "k candidate: at the range ends, not a candidate");
        check(t::k_mult_candidate(1.4, 1.25, 1, m) && near(m, 1.5, 1e-9), "k candidate: clamped to 1.5 (still a candidate from 1.4)");
        check(!t::k_mult_candidate(1.0, 0.0, 1, m), "k candidate: a non-positive step is not a candidate");
        // explore veto: the candidate's explore flips (20/40, lower bound ~0.37) above the base's upper bound (2/40, ~0.14) and the control's
        check(t::explore_veto(2, 40, 20, 40, 0.074, 16), "explore veto: the candidate's explore flips above noise block the move");
        check(!t::explore_veto(2, 40, 5, 40, 0.074, 16), "explore veto: the candidate at the base's rate passes");
        check(!t::explore_veto(2, 40, 20, 10, 0.074, 16), "explore veto: fewer than 16 candidate explore cuts: not blocked");
    }
    {
        // the margin's state survives a save and a load; an out-of-range m is refused
        g::State s;
        s.tune.k_mult = 1.125;
        s.tune.k_step = 1.5625f;
        s.tune.k_last_win = 1;
        s.tune.k_first = -1;
        s.tune.k_stage = 1;
        g::State u;
        check(g::parse(g::serialize(s, 12), 12, u) && near(u.tune.k_mult, 1.125, 1e-9) && near(u.tune.k_step, 1.5625, 1e-6) &&
                  u.tune.k_last_win == 1 && u.tune.k_first == -1 && u.tune.k_stage == 1,
              "round trip: the margin m, its step and direction come back");
        g::State v;
        check(!g::parse("fingerprint 0c\ntune_k 2.0 1.25 0 1 0\n", 12, v), "tune: a margin above 1.5 is refused");
    }

    std::printf("recheck_gate_test: ok\n");
    return 0;
}
