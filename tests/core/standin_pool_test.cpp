// STRATA_STANDIN_LEARN's pooled estimates (include/strata/core/standin_pool.hpp), host only: the shrinkage, the readiness rule,
// the parent groups rebuilt at load, the old table file's read, and the balance step.  With a path argument (a COPY of the live
// table, e.g. data/standin-learn-*.bin copied to a temp dir) the same checks run on the real pairs, read only.
#include "strata/core/standin_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace sdp = strata::standin;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (cond) { ++g_pass; }                           \
        else {                                            \
            ++g_fail;                                     \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);            \
            std::fprintf(stderr, "\n");                   \
        }                                                 \
    } while (0)

// one scalar sample: y_s = 1, y_e = ratio (so the pair's c is the ratio)
static void add_ratio(sdp::Pair& p, double ratio) {
    p.num = sdp::kDecay * p.num + ratio;
    p.den = sdp::kDecay * p.den + 1.0;
    p.ee = sdp::kDecay * p.ee + ratio * ratio;
    ++p.n;
}
static void add_ratio_group(sdp::Sums& g, double ratio) { sdp::add_sample(g, ratio, 1.0, ratio * ratio); }

static void test_shrinkage() {
    // a pair with one sample (c = 1.0) inside a parent group of 100 samples (c = 1.5): the parent's c, within 0.05
    sdp::Pair one;
    add_ratio(one, 1.0);
    sdp::Sums layer;
    for (int i = 0; i < 100; ++i) add_ratio_group(layer, 1.5);
    const sdp::Est a = sdp::estimate(one, nullptr, &layer);
    CHECK(a.pooled && a.ready, "one-sample pair with a ready layer parent: pooled %d ready %d", a.pooled, a.ready);
    CHECK(std::fabs(a.c - 1.5) < 0.05, "one-sample pair takes the parent's c: got %.4f want ~1.5", a.c);
    CHECK(std::fabs(a.c - 1.0) > 0.4, "one-sample pair moved toward the parent: got %.4f", a.c);

    // 1,000 own samples at c = 1.0 with the same parent: its own c (within 0.01)
    sdp::Pair many;
    for (int i = 0; i < 1000; ++i) add_ratio(many, 1.0);
    const sdp::Est b = sdp::estimate(many, nullptr, &layer);
    CHECK(std::fabs(b.c - 1.0) < 0.01, "1000-sample pair keeps its own c: got %.4f want 1.0", b.c);

    // no parent: the pair's own estimate, as before pooling (c = its own num / den)
    const sdp::Est c0 = sdp::estimate(one, nullptr, nullptr);
    CHECK(!c0.pooled && !c0.ready && std::fabs(c0.c - 1.0) < 1e-12, "no parent, one sample: not ready, own c (%d %d %.4f)", c0.pooled,
          c0.ready, c0.c);
    // the err of a perfect pair (y_s = y_e) is 0
    const sdp::Est perfect = sdp::estimate(many, nullptr, nullptr);
    CHECK(std::fabs(perfect.err) < 1e-6, "err of a perfect pair (no parent) is 0: got %.6f", perfect.err);
}

static void test_readiness_and_parent_choice() {
    sdp::Pair p;
    add_ratio(p, 1.0);
    sdp::Sums g31, g32;
    for (int i = 0; i < 31; ++i) add_ratio_group(g31, 1.5);
    for (int i = 0; i < 32; ++i) add_ratio_group(g32, 1.5);
    CHECK(!sdp::estimate(p, nullptr, &g31).ready, "a 31-sample parent is not a parent");
    CHECK(sdp::estimate(p, nullptr, &g32).ready, "a 32-sample parent makes the pair ready");

    sdp::Pair own;
    for (int i = 0; i < 31; ++i) add_ratio(own, 1.0);
    CHECK(!sdp::estimate(own, nullptr, nullptr).ready, "31 own samples, no parent: not ready");
    add_ratio(own, 1.0);
    CHECK(sdp::estimate(own, nullptr, nullptr).ready, "32 own samples: ready");

    // (layer, e) group preferred over the layer; a short (layer, e) group falls back to the layer
    sdp::Sums expert, layerB;
    for (int i = 0; i < 32; ++i) add_ratio_group(expert, 1.2);
    for (int i = 0; i < 32; ++i) add_ratio_group(layerB, 1.8);
    const sdp::Est ex = sdp::estimate(p, &expert, &layerB);
    CHECK(std::fabs(ex.c - 1.2) < 0.05, "expert group is the parent: c %.4f want ~1.2", ex.c);
    sdp::Sums shortExpert;
    for (int i = 0; i < 31; ++i) add_ratio_group(shortExpert, 1.2);
    const sdp::Est fb = sdp::estimate(p, &shortExpert, &layerB);
    CHECK(std::fabs(fb.c - 1.8) < 0.1, "short expert group falls back to the layer: c %.4f want ~1.8", fb.c);
}

static void test_rebuild_groups() {
    std::unordered_map<uint64_t, sdp::Pair> pairs;
    sdp::Pair a, b, c;
    for (int i = 0; i < 40; ++i) add_ratio(a, 1.0);
    for (int i = 0; i < 5; ++i) add_ratio(b, 1.0);
    for (int i = 0; i < 7; ++i) add_ratio(c, 1.3);
    pairs[sdp::pair_key(0, 5, 9)] = a;
    pairs[sdp::pair_key(0, 5, 10)] = b;
    pairs[sdp::pair_key(0, 6, 1)] = c;
    std::vector<sdp::Sums> layer(3);
    std::unordered_map<uint64_t, sdp::Sums> exp;
    sdp::rebuild_groups(pairs, layer, exp);
    CHECK(layer[0].n == 52, "layer 0 raw samples 52: got %lld", (long long) layer[0].n);
    CHECK(layer[1].n == 0 && layer[2].n == 0, "layers without pairs stay empty");
    CHECK(exp.count(sdp::expert_key(0, 5)) && exp[sdp::expert_key(0, 5)].n == 45, "expert (0,5) has 45 samples");
    CHECK(exp.count(sdp::expert_key(0, 6)) && exp[sdp::expert_key(0, 6)].n == 7, "expert (0,6) has 7 samples");
    const double wantW = sdp::decayed_weight(40) + sdp::decayed_weight(5) + sdp::decayed_weight(7);
    CHECK(std::fabs(layer[0].w - wantW) < 1e-9, "layer weight is the pairs' decayed weights");
    CHECK(std::fabs(exp[sdp::expert_key(0, 5)].w - (sdp::decayed_weight(40) + sdp::decayed_weight(5))) < 1e-9,
          "expert weight is the decayed weight of its pairs");
    // a single unit sample has weight 1 (the group's add_sample agrees with decayed_weight)
    sdp::Sums one;
    sdp::add_sample(one, 1.0, 1.0, 1.0);
    CHECK(std::fabs(one.w - sdp::decayed_weight(1)) < 1e-12, "add_sample weight matches decayed_weight(1)");
}

// the old file format, written the way the saver wrote it
static std::vector<uint8_t> make_old_table(int64_t layers, const std::vector<std::pair<uint64_t, sdp::Pair>>& recs, uint32_t declared) {
    std::vector<uint8_t> out(16);
    const uint32_t hd[4] = {sdp::kMagic, (uint32_t) layers, 512u, declared};
    std::memcpy(out.data(), hd, 16);
    for (const auto& kv : recs) {
        const size_t at = out.size();
        out.resize(at + 40);
        const uint32_t n = (uint32_t) kv.second.n;
        std::memcpy(out.data() + at, &kv.first, 8);
        std::memcpy(out.data() + at + 8, &kv.second.num, 8);
        std::memcpy(out.data() + at + 16, &kv.second.den, 8);
        std::memcpy(out.data() + at + 24, &kv.second.ee, 8);
        std::memcpy(out.data() + at + 32, &n, 4);
        std::memcpy(out.data() + at + 36, &kv.second.c, 4);
    }
    return out;
}

static void test_old_table_loads() {
    sdp::Pair p;
    add_ratio(p, 1.25);
    p.c = 1.25f;
    const uint64_t k1 = sdp::pair_key(3, 100, 7), k2 = sdp::pair_key(3, 511, 511);
    // a record with an out-of-range expert is skipped; the rest load
    const sdp::Pair bad = p;
    const uint64_t kbad = sdp::pair_key(3, 512, 1);
    std::vector<uint8_t> buf = make_old_table(47, {{k1, p}, {kbad, bad}, {k2, p}}, 3);
    std::vector<sdp::Loaded> out;
    CHECK(sdp::parse_table(buf.data(), buf.size(), 47, out), "old-format table parses");
    CHECK(out.size() == 2, "out-of-range record skipped: got %zu records", out.size());
    CHECK(!out.empty() && out[0].key == k1 && out[0].p.n == 1 && std::fabs(out[0].p.c - 1.25f) < 1e-6 && std::fabs(out[0].p.num - p.num) < 1e-12,
          "fields round-trip");
    // a different model's layer count is refused
    std::vector<sdp::Loaded> out2;
    CHECK(!sdp::parse_table(buf.data(), buf.size(), 48, out2), "a table of another layer count is refused");
    // a file cut short keeps its whole records
    std::vector<uint8_t> cut(buf.begin(), buf.end() - 10);
    std::vector<sdp::Loaded> out3;
    CHECK(sdp::parse_table(cut.data(), cut.size(), 47, out3) && out3.size() == 1, "truncated last record is skipped (got %zu)", out3.size());
    // not a table at all
    std::vector<uint8_t> junk(64, 0x11);
    std::vector<sdp::Loaded> out4;
    CHECK(!sdp::parse_table(junk.data(), junk.size(), 47, out4), "junk is refused");
}

// The balance step: 12 experts in one layer, values from their errs; top decile +1, bottom -1; the two-round rule.
static void test_balance_step() {
    const int64_t layers = 1;
    std::vector<double> best(512, -1.0);
    std::vector<float> use(512, 1.0f), miss(512, 0.0f);
    for (int e = 0; e < 12; ++e) best[(size_t) e] = 0.1 + 0.01 * e;   // expert 11 has the largest err (value)
    std::vector<int8_t> state(512, 0);
    std::vector<uint8_t> opp(512, 0);
    sdp::balance_step(state, opp, best, use, miss, layers);
    CHECK(state[11] == 1 && state[0] == -1, "round 1: the top expert kept (%d), the bottom given up (%d)", state[11], state[0]);
    CHECK(state[5] == 0, "middle experts unchanged");
    int plus = 0, minus = 0;
    for (int e = 0; e < 512; ++e) { plus += state[(size_t) e] > 0; minus += state[(size_t) e] < 0; }
    CHECK(plus == 1 && minus == 1, "one kept and one given up in a 12-expert layer (k = 1): %d %d", plus, minus);
    // the values reversed: expert 11 is now the bottom. Its +1 holds for one more round (op 1), then goes to 0 (op 2)
    for (int e = 0; e < 12; ++e) best[(size_t) e] = 0.1 + 0.01 * (11 - e);
    sdp::balance_step(state, opp, best, use, miss, layers);
    CHECK(state[11] == 1, "after 1 opposite round the state holds (+1)");
    sdp::balance_step(state, opp, best, use, miss, layers);
    CHECK(state[11] == 0, "after 2 opposite rounds the state goes to 0 (got %d)", state[11]);
    // fewer than 10 ranked experts: no decision
    std::vector<double> few(512, -1.0);
    for (int e = 0; e < 9; ++e) few[(size_t) e] = 0.5;
    std::vector<int8_t> st2(512, 0);
    std::vector<uint8_t> op2(512, 0);
    sdp::balance_step(st2, op2, few, use, miss, layers);
    bool any = false;
    for (const int8_t s : st2) any = any || s != 0;
    CHECK(!any, "9 ranked experts make no decision");
}

// The live table copy (argv[1]): the old file loads, pooling makes more pairs ready than own counts, and the balance ranks
// >= 10 experts in the layers that have data and keeps / gives up the top / bottom decile.  Read only.
static int test_live(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 16) { std::fprintf(stderr, "short file\n"); return 1; }
    uint32_t hd[4] = {0, 0, 0, 0};
    std::memcpy(hd, buf.data(), 16);
    const int64_t layers = (int64_t) hd[1];
    std::vector<sdp::Loaded> recs;
    CHECK(sdp::parse_table(buf.data(), buf.size(), layers, recs), "live table parses (layers %lld, declared %u)", (long long) layers, hd[3]);
    std::unordered_map<uint64_t, sdp::Pair> pairs;
    for (const auto& r : recs) pairs[r.key] = r.p;
    std::vector<sdp::Sums> layer_g((size_t) layers);
    std::unordered_map<uint64_t, sdp::Sums> exp_g;
    sdp::rebuild_groups(pairs, layer_g, exp_g);
    int own = 0, pooled = 0;
    std::vector<double> best((size_t) layers * 512, -1.0);
    for (const auto& kv : pairs) {
        const auto it = exp_g.find(sdp::expert_key((int64_t) (kv.first >> 40), (int64_t) ((kv.first >> 20) & 0xFFFFFu)));
        const sdp::Sums* eg = it == exp_g.end() ? nullptr : &it->second;
        const sdp::Sums* lg = &layer_g[(size_t) (kv.first >> 40)];
        const sdp::Est es = sdp::estimate(kv.second, eg, lg);
        if (kv.second.n >= sdp::kReady) ++own;
        if (!es.ready) continue;
        ++pooled;
        const size_t i = (size_t) (kv.first >> 40) * 512 + (size_t) ((kv.first >> 20) & 0xFFFFFu);
        if (best[i] < 0.0 || es.err < best[i]) best[i] = es.err;
    }
    std::printf("live: %zu pairs, own ready %d, pooled ready %d, experts pooled %zu\n", pairs.size(), own, pooled, exp_g.size());
    CHECK(pooled >= own, "pooling never loses a ready pair: %d < %d", pooled, own);
    std::vector<float> use((size_t) layers * 512, 1.0f), miss((size_t) layers * 512, 0.0f);
    std::vector<int8_t> state((size_t) layers * 512, 0);
    std::vector<uint8_t> opp((size_t) layers * 512, 0);
    sdp::balance_step(state, opp, best, use, miss, layers);
    int layers_ranked = 0, bad = 0;
    for (int64_t l = 0; l < layers; ++l) {
        int ranked = 0, plus = 0, minus = 0;
        for (int e = 0; e < 512; ++e) {
            if (best[(size_t) (l * 512 + e)] < 0.0) continue;
            ++ranked;
            plus += state[(size_t) (l * 512 + e)] > 0;
            minus += state[(size_t) (l * 512 + e)] < 0;
        }
        if (ranked < 10) continue;
        ++layers_ranked;
        const int k = std::max(1, ranked / 10);
        if (plus != k || minus != k) ++bad;
    }
    std::printf("live: %d layers with >= 10 ranked experts\n", layers_ranked);
    CHECK(layers_ranked >= 1, "the balance ranks >= 10 experts in some layer with data");
    CHECK(bad == 0, "every ranked layer keeps and gives up one decile (%d bad)", bad);
    return g_fail == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    test_shrinkage();
    test_readiness_and_parent_choice();
    test_rebuild_groups();
    test_old_table_loads();
    test_balance_step();
    if (argc > 1) test_live(argv[1]);
    else std::printf("live table check skipped (no path given)\n");
    std::printf("standin_pool_test: %d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
