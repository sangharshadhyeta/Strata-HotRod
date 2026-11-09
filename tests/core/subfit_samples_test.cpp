// tests/core/subfit_samples_test.cpp - SPEC-substitute-fit stage 1a: the sample file's format, CPU only.
// Covers: bf16 rounding, the env switches, a round trip of both record tags (also at distinct layers with distinct x, y_s, y_e: a
// swap or a zeroed field fails it), the header check on append, the size cap, and that a writer which was never opened writes nothing.
#include "strata/core/subfit_samples.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

using strata::core::subfit::Record;
using strata::core::subfit::SampleReader;
using strata::core::subfit::SampleWriter;

namespace {

int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

std::string tmp_path(const char* tag) {
    const auto dir = std::filesystem::temp_directory_path();
    return (dir / (std::string("subfit_test_") + tag + "_" + std::to_string((long long) std::hash<std::string>{}(tag)) + ".bin")).string();
}

// the value bf16 keeps for v (what the file holds after a round trip)
float bf(float v) { return strata::core::subfit::bf16_to_f32(strata::core::subfit::f32_to_bf16(v)); }

void test_bf16() {
    using namespace strata::core::subfit;
    CHECK(bf(1.0f) == 1.0f, "1.0 is exact in bf16");
    CHECK(bf(-2.0f) == -2.0f, "-2.0 is exact in bf16");
    CHECK(bf(0.0f) == 0.0f, "0 is exact");
    bool rel_ok = true;
    for (int i = 1; i < 2000; ++i) {
        const float v = std::sin((float) i) * std::pow(10.0f, (float) (i % 7) - 3.0f);
        if (std::fabs(bf(v) - v) > std::fabs(v) * (1.0f / 256.0f)) rel_ok = false;   // round to nearest: at most half an ulp of 8 bits
    }
    CHECK(rel_ok, "bf16 round trip stays within 2^-8 relative");
    CHECK(std::isinf(bf(INFINITY)) && bf(INFINITY) > 0, "+inf survives");
    CHECK(std::isnan(bf(NAN)), "NaN stays NaN");
    // ties to even: 1 + 2^-8 is halfway between 1 and 1 + 2^-7; the even mantissa (1.0) wins
    CHECK(bf(1.0f + 1.0f / 256.0f) == 1.0f, "ties round to even");
}

void test_env() {
    using namespace strata::core::subfit;
    CHECK(!log_on_env(nullptr), "unset: logger off");
    CHECK(!log_on_env("0"), "0: logger off");
    CHECK(!log_on_env(""), "empty: logger off");
    CHECK(log_on_env("1"), "1: logger on");
    CHECK(probe_every_env(nullptr) == 64, "probe default 64");
    CHECK(probe_every_env("16") == 16, "probe 16");
    CHECK(probe_every_env("0") == 64, "probe 0 falls back to the default");
    CHECK(probe_every_env("-3") == 64, "probe negative falls back to the default");
}

void test_round_trip() {
    using namespace strata::core::subfit;
    const uint32_t N = 16, L = 4;
    const std::string path = tmp_path("roundtrip");
    std::filesystem::remove(path);
    std::string err;
    {
        SampleWriter w;
        CHECK(w.open(path, N, L, err), ("open: " + err).c_str());
        std::vector<float> x(N), ys(N), ye(N);
        for (uint32_t i = 0; i < N; ++i) {
            x[i] = 0.25f * (float) i - 1.5f;
            ys[i] = std::sin((float) i);
            ye[i] = std::cos((float) i) * 3.0f;
        }
        CHECK(w.write_stand_in(2, 300, 17, 5, x.data(), ys.data(), ye.data()), "tag 1 written");
        CHECK(w.write_probe(3, 511, 9, 0.375f, x.data(), ye.data()), "tag 2 written");
        CHECK(w.take_request_count() == 2, "two records this request");
        CHECK(w.take_request_count() == 0, "the request count resets");
        CHECK(w.samples_total() == 2, "two records in total");
    }
    {
        SampleReader r;
        CHECK(r.open(path, err), ("reopen: " + err).c_str());
        CHECK(r.header().n_embd == N && r.header().n_layers == L && r.header().version == kVersion, "header fields");
        Record a, b, c;
        CHECK(r.next(a, err), "first record");
        CHECK(a.tag == 1 && a.layer == 2 && a.e == 300 && a.s == 17 && a.rank == 5, "tag 1 fields");
        CHECK(a.x.size() == N && a.ys.size() == N && a.ye.size() == N, "tag 1 vector sizes");
        bool same = true;
        for (uint32_t i = 0; i < N; ++i) {
            const float xv = 0.25f * (float) i - 1.5f;
            if (a.x[i] != bf(xv) || a.ys[i] != bf(std::sin((float) i)) || a.ye[i] != bf(std::cos((float) i) * 3.0f)) same = false;
        }
        CHECK(same, "tag 1 values are the bf16 of the input");
        CHECK(r.next(b, err), "second record");
        CHECK(b.tag == 2 && b.layer == 3 && b.e == 511 && b.rank == 9 && b.w == 0.375f, "tag 2 fields and exact w");
        CHECK(b.x.size() == N && b.ye.size() == N && b.ys.empty(), "tag 2 has x and y_e only");
        CHECK(!r.next(c, err), "end of file");
        CHECK(err.empty(), "clean end of file has no error");
    }
    std::filesystem::remove(path);
}

void test_append_and_header() {
    using namespace strata::core::subfit;
    const std::string path = tmp_path("append");
    std::filesystem::remove(path);
    std::string err;
    std::vector<float> v(8, 1.0f);
    {
        SampleWriter w;
        CHECK(w.open(path, 8, 2, err), "first open");
        CHECK(w.write_probe(0, 1, 0, 1.0f, v.data(), v.data()), "write");
    }
    {
        SampleWriter w;
        CHECK(w.open(path, 8, 2, err), "append to a matching file");
        CHECK(w.write_probe(1, 2, 3, 2.0f, v.data(), v.data()), "write after append");
    }
    {
        SampleWriter w;
        CHECK(!w.open(path, 16, 2, err), "a file of another width is refused");
        CHECK(!err.empty(), "the refusal names the problem");
        CHECK(!w.write_probe(0, 0, 0, 0.0f, v.data(), v.data()), "a refused writer writes nothing");
    }
    {
        SampleReader r;
        CHECK(r.open(path, err), "read back");
        Record rec;
        int n = 0;
        while (r.next(rec, err)) ++n;
        CHECK(n == 2, "two records after the append");
    }
    std::filesystem::remove(path);
}

void test_cap_and_never_opened() {
    using namespace strata::core::subfit;
    const uint32_t N = 16;
    const std::string path = tmp_path("cap");
    std::filesystem::remove(path);
    std::string err;
    std::vector<float> v(N, 0.5f);
    SampleWriter w;
    w.set_cap(16 + 10 + 4 * N);   // the header and exactly one probe record (tag, layer, e, rank, w, x and y_e as bf16)
    CHECK(w.open(path, N, 1, err), "open with a small cap");
    CHECK(w.write_probe(0, 1, 0, 1.0f, v.data(), v.data()), "the first record fits");
    CHECK(!w.write_probe(0, 2, 1, 1.0f, v.data(), v.data()), "the second would pass the cap");
    CHECK(w.capped(), "the writer reports the cap");
    CHECK(!w.write_probe(0, 3, 2, 1.0f, v.data(), v.data()), "nothing is written after the cap");
    CHECK(w.samples_total() == 1, "one record in total");
    CHECK(std::filesystem::file_size(path) <= 16 + 10 + 4 * N, "the file stays within the cap");
    w.close();
    std::filesystem::remove(path);

    // switch off: a writer that is never opened has no file and writes nothing
    SampleWriter off;
    CHECK(!off.is_open(), "never opened");
    CHECK(!off.write_stand_in(0, 0, 0, 0, v.data(), v.data(), v.data()), "a writer that is not open writes nothing");
    CHECK(!std::filesystem::exists(path), "no file is created without an open");
}

// Step 0 of stage 2: every record's x, y_s and y_e must come back as their own inputs, at distinct layers, with no swap and no zeroing.
// Records of several layers (0, 5, 47 of the 48-layer pack), distinct non-zero vectors per record and per field; the stored y_e of each
// tag-1 record must be non-zero, so a zeroed or swapped field fails the exact comparison below.
void test_round_trip_distinct() {
    using namespace strata::core::subfit;
    const uint32_t N = 2560, L = 48;
    const std::string path = tmp_path("distinct");
    std::filesystem::remove(path);
    std::string err;
    const uint16_t layers[] = {0, 5, 47, 0};
    auto fill = [&](std::vector<float>& v, int rec, int field) {
        v.assign(N, 0.0f);
        for (uint32_t i = 0; i < N; ++i)   // distinct per (record, field, index); never zero
            v[i] = 0.001f * (float) (i + 1) * (float) (rec + 1) + 0.5f * (float) field + (i % 3 == 0 ? -0.25f : 0.125f);
    };
    std::vector<float> x, ys, ye;
    {
        SampleWriter w;
        CHECK(w.open(path, N, L, err), ("open: " + err).c_str());
        for (int r = 0; r < 4; ++r) {
            fill(x, r, 0);
            fill(ys, r, 1);
            fill(ye, r, 2);
            if (r == 2) {
                CHECK(w.write_probe(layers[r], (uint16_t) (400 + r), (uint8_t) r, 0.25f + 0.0625f * (float) r, x.data(), ye.data()),
                      "tag 2 written");
            } else {
                CHECK(w.write_stand_in(layers[r], (uint16_t) (100 + r), (uint16_t) (200 + r), (uint8_t) (r + 2), x.data(), ys.data(), ye.data()),
                      "tag 1 written");
            }
        }
    }
    SampleReader rd;
    CHECK(rd.open(path, err), ("reopen: " + err).c_str());
    for (int r = 0; r < 4; ++r) {
        Record got;
        CHECK(rd.next(got, err), "a record is read");
        if (r == 2) {
            CHECK(got.tag == 2 && got.layer == layers[r] && got.e == 402 && got.rank == 2, "tag 2 fields as written");
            CHECK(got.w == 0.25f + 0.0625f * 2.0f, "tag 2 w exact");
            fill(x, r, 0);
            fill(ye, r, 2);
            bool ok = got.x.size() == N && got.ye.size() == N && got.ys.empty();
            for (uint32_t i = 0; ok && i < N; ++i) ok = got.x[i] == bf(x[i]) && got.ye[i] == bf(ye[i]);
            CHECK(ok, "tag 2: x and y_e are the bf16 of their own inputs (no swap, no zeroing)");
            continue;
        }
        CHECK(got.tag == 1 && got.layer == layers[r] && got.e == 100 + r && got.s == 200 + r && got.rank == r + 2,
              "tag 1 fields as written (layer, e, s, rank)");
        fill(x, r, 0);
        fill(ys, r, 1);
        fill(ye, r, 2);
        bool ok = got.x.size() == N && got.ys.size() == N && got.ye.size() == N;
        for (uint32_t i = 0; ok && i < N; ++i) ok = got.x[i] == bf(x[i]) && got.ys[i] == bf(ys[i]) && got.ye[i] == bf(ye[i]);
        CHECK(ok, "tag 1: x, y_s and y_e are the bf16 of their own inputs (no swap, no zeroing)");
        double ye_energy = 0.0;
        for (float v : got.ye) ye_energy += (double) v * (double) v;
        CHECK(ye_energy > 1.0, "tag 1: the stored y_e is not zero");
    }
    Record tail;
    CHECK(!rd.next(tail, err) && err.empty(), "clean end after four records");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    test_bf16();
    test_env();
    test_round_trip();
    test_round_trip_distinct();
    test_append_and_header();
    test_cap_and_never_opened();
    if (failures == 0) {
        std::printf("subfit_samples_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "subfit_samples_test: %d check(s) failed\n", failures);
    return 1;
}
