// Test of the KV-streaming page pool kernels (docs/kv-streaming-s3d-design.md, step S3d-2).
//
// The CUDA backend exposes its append/resolve/copy step through a proc address (ggml_backend_kv_stream_run) and a
// pinned-host buffer type (ggml_backend_kv_host_buffer_type). The test drives them with random, locality-heavy
// selections and appended rows and, after every call, checks the properties the design promises - not which slot the
// parallel sweep happened to pick:
//   * every page named by the call is resident, and no page claimed by the call is left half done (-2)
//   * page table and slot table agree (a bijection between resident pages and used slots)
//   * every resident slot equals its page in the host tensors, byte for byte (coherence, including rows appended to
//     a page that was already resident)
//   * the number of pages fetched equals the number of requested pages that were not resident before the call
//   * pages used by the call were never evicted by it, and carry this call's epoch
//   * the 64-bit counters agree with a model of the same events
// and the edge cases: a call that selects nothing, and a call that names more distinct pages than the pool holds.
//
// A backend without a KV-stream pool (CPU-only builds) makes the test skip.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

typedef ggml_backend_buffer_type_t (*kv_host_buft_fn_t)(void);

typedef int (*kv_stream_run_fn_t)(
        ggml_backend_t backend, int32_t * state, int64_t n_pages, int64_t n_slots, int page, int64_t n_cells,
        const void * host_k, const void * host_v, void * pool_k, void * pool_v, size_t row_bytes_k, size_t row_bytes_v,
        const int32_t * idxs, int n_idxs, const int32_t * lists, const int32_t * counts, int n_lists, int list_stride);

static const int     PAGE    = 4;
static const int64_t N_CELLS = 39998;                      // not a multiple of PAGE: the last page is cut short
static const int64_t N_PAGES = (N_CELLS + PAGE - 1)/PAGE;  // 10000
static const int64_t N_SLOTS = 1024;                       // the minimum the kernels accept
static const int     DIM     = 64;                         // f16 values per row
static const size_t  ROWB    = DIM*sizeof(uint16_t);       // 128 bytes, a multiple of 16
static const int     MAXL    = 4;                          // lists per call
static const int     STRIDE  = 4608;                       // cell capacity of a list (>= 1152 pages)
static const int     MAXI    = 64;                         // appended cells per call

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            return false; \
        } \
    } while (0)

struct kv_test {
    ggml_backend_t     backend = nullptr;
    kv_stream_run_fn_t run     = nullptr;

    ggml_context *        ctx_host = nullptr;
    ggml_context *        ctx_dev  = nullptr;
    ggml_backend_buffer_t buf_host = nullptr;
    ggml_backend_buffer_t buf_dev  = nullptr;

    ggml_tensor * host_k  = nullptr;
    ggml_tensor * host_v  = nullptr;
    ggml_tensor * pool_k  = nullptr;
    ggml_tensor * pool_v  = nullptr;
    ggml_tensor * state   = nullptr;
    ggml_tensor * lists_t = nullptr;
    ggml_tensor * count_t = nullptr;
    ggml_tensor * idxs_t  = nullptr;

    std::vector<uint16_t> hk; // the CPU copy of the host tensors
    std::vector<uint16_t> hv;

    std::vector<int32_t> table_before; // page -> slot before the current call
    uint64_t t_fetched = 0;
    uint64_t t_lookups = 0;
    uint64_t t_calls   = 0;
    int      epoch     = 0;

    int64_t state_ints() const { return ggml_kv_stream_state_ints(N_PAGES, N_SLOTS); }

    bool init(ggml_backend_buffer_type_t host_buft) {
        ggml_init_params ip = { /*.mem_size =*/ 16*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };

        ctx_host = ggml_init(ip);
        ctx_dev  = ggml_init(ip);

        host_k = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F16, DIM, N_CELLS);
        host_v = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F16, DIM, N_CELLS);

        pool_k  = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_F16, DIM, N_SLOTS*PAGE);
        pool_v  = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_F16, DIM, N_SLOTS*PAGE);
        state   = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_I32, state_ints());
        lists_t = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_I32, (int64_t) MAXL*STRIDE);
        count_t = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_I32, MAXL);
        idxs_t  = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_I32, MAXI);

        buf_host = ggml_backend_alloc_ctx_tensors_from_buft(ctx_host, host_buft);
        buf_dev  = ggml_backend_alloc_ctx_tensors(ctx_dev, backend);
        if (!buf_host || !buf_dev) {
            fprintf(stderr, "failed to allocate the test buffers\n");
            return false;
        }

        std::mt19937 rng(1234);
        hk.resize((size_t) N_CELLS*DIM);
        hv.resize((size_t) N_CELLS*DIM);
        for (auto & x : hk) { x = (uint16_t) rng(); }
        for (auto & x : hv) { x = (uint16_t) rng(); }
        ggml_backend_tensor_set(host_k, hk.data(), 0, hk.size()*sizeof(uint16_t));
        ggml_backend_tensor_set(host_v, hv.data(), 0, hv.size()*sizeof(uint16_t));

        reset_state();
        return true;
    }

    // the initial state: page table and slot_page -1 (none), everything else 0 (what llama_kv_cache::stream_reset writes)
    void reset_state() {
        ggml_backend_tensor_memset(state, 0x00, 0, ggml_nbytes(state));
        ggml_backend_tensor_memset(state, 0xFF, 0, (size_t) (N_PAGES + N_SLOTS)*sizeof(int32_t));

        table_before.assign((size_t) N_PAGES, -1);
        t_fetched = t_lookups = t_calls = 0;
        epoch = 0;
    }

    void write_row(int64_t cell, std::mt19937 & rng) {
        for (int d = 0; d < DIM; ++d) {
            hk[(size_t) cell*DIM + d] = (uint16_t) rng();
            hv[(size_t) cell*DIM + d] = (uint16_t) rng();
        }
        ggml_backend_tensor_set(host_k, &hk[(size_t) cell*DIM], (size_t) cell*ROWB, ROWB);
        ggml_backend_tensor_set(host_v, &hv[(size_t) cell*DIM], (size_t) cell*ROWB, ROWB);
    }

    // one streamed step, then the checks
    bool step(const std::vector<std::vector<int32_t>> & lists, const std::vector<int32_t> & idxs, bool expect_overflow) {
        CHECK((int) lists.size() <= MAXL && (int) idxs.size() <= MAXI, "test setup");

        std::vector<int32_t> flat((size_t) MAXL*STRIDE, -1);
        std::vector<int32_t> counts(MAXL, 0);
        for (size_t l = 0; l < lists.size(); ++l) {
            CHECK((int) lists[l].size() <= STRIDE, "list too long");
            std::copy(lists[l].begin(), lists[l].end(), flat.begin() + (int64_t) l*STRIDE);
            counts[l] = (int32_t) lists[l].size();
        }
        ggml_backend_tensor_set(lists_t, flat.data(), 0, flat.size()*sizeof(int32_t));
        ggml_backend_tensor_set(count_t, counts.data(), 0, counts.size()*sizeof(int32_t));
        if (!idxs.empty()) {
            ggml_backend_tensor_set(idxs_t, idxs.data(), 0, idxs.size()*sizeof(int32_t));
        }

        const int rc = run(backend, (int32_t *) state->data, N_PAGES, N_SLOTS, PAGE, N_CELLS,
                host_k->data, host_v->data, pool_k->data, pool_v->data, ROWB, ROWB,
                (const int32_t *) idxs_t->data, (int) idxs.size(),
                (const int32_t *) lists_t->data, (const int32_t *) count_t->data, (int) lists.size(), STRIDE);
        CHECK(rc == 0, "the kernel step failed");

        return verify(lists, expect_overflow);
    }

    bool verify(const std::vector<std::vector<int32_t>> & lists, bool expect_overflow) {
        std::vector<int32_t> st((size_t) state_ints());
        ggml_backend_tensor_get(state, st.data(), 0, st.size()*sizeof(int32_t));

        const int32_t * page_table = st.data();
        const int32_t * slot_page  = page_table + N_PAGES;
        const int32_t * slot_stamp = slot_page  + N_SLOTS;
        const int32_t * ctl        = slot_stamp + 4*N_SLOTS; // after slot_stamp, slot_ref, miss_page, miss_slot

        // the pages this call named
        std::set<int32_t> requested;
        for (const auto & list : lists) {
            for (int32_t c : list) {
                if (c >= 0 && c < N_CELLS) {
                    requested.insert(c/PAGE);
                }
            }
        }

        // structure: no claim left over, page table and slot table agree
        int64_t n_res = 0;
        for (int64_t p = 0; p < N_PAGES; ++p) {
            const int32_t s = page_table[p];
            CHECK(s >= -1, "page %lld left claimed (%d)", (long long) p, s);
            if (s >= 0) {
                CHECK(s < N_SLOTS, "page %lld points at slot %d", (long long) p, s);
                CHECK(slot_page[s] == p, "page %lld -> slot %d -> page %d", (long long) p, s, slot_page[s]);
                ++n_res;
            }
        }
        int64_t n_used = 0;
        for (int64_t s = 0; s < N_SLOTS; ++s) {
            const int32_t p = slot_page[s];
            if (p >= 0) {
                CHECK(p < N_PAGES && page_table[p] == s, "slot %lld -> page %d -> slot %d", (long long) s, p, p < N_PAGES ? page_table[p] : -99);
                ++n_used;
            }
        }
        CHECK(n_res == n_used, "%lld resident pages but %lld used slots", (long long) n_res, (long long) n_used);

        // the events of this call, by the model
        int64_t expect_fetch = 0;
        for (int32_t p : requested) {
            if (table_before[(size_t) p] < 0) {
                ++expect_fetch;
            }
        }

        if (!lists.empty()) {
            ++epoch;
            ++t_calls;
        }

        if (expect_overflow) {
            CHECK(ctl[3] == 1, "the overflow flag is not set");
            CHECK(n_res == N_SLOTS, "%lld resident pages after overflow, expected the full pool", (long long) n_res);
            for (int64_t s = 0; s < N_SLOTS; ++s) {
                CHECK(requested.count(slot_page[s]) == 1, "slot %lld holds a page nobody asked for", (long long) s);
            }
        } else {
            CHECK(ctl[3] == 0, "unexpected overflow flag");
            for (int32_t p : requested) {
                CHECK(page_table[p] >= 0, "requested page %d is not resident", p);
                CHECK(slot_stamp[page_table[p]] == epoch, "slot of page %d carries epoch %d, expected %d", p, slot_stamp[page_table[p]], epoch);
            }
            if (!lists.empty()) {
                CHECK(ctl[2] == expect_fetch, "fetched %d pages, expected %lld", ctl[2], (long long) expect_fetch);
            }
            t_fetched += (uint64_t) expect_fetch;
            t_lookups += (uint64_t) requested.size();
        }

        if (!lists.empty() && !expect_overflow) {
            CHECK(ctl[0] == epoch, "epoch %d, expected %d", ctl[0], epoch);
            const uint64_t fetched = (uint64_t) (uint32_t) ctl[4] | ((uint64_t) (uint32_t) ctl[5] << 32);
            const uint64_t lookups = (uint64_t) (uint32_t) ctl[6] | ((uint64_t) (uint32_t) ctl[7] << 32);
            const uint64_t calls   = (uint64_t) (uint32_t) ctl[8] | ((uint64_t) (uint32_t) ctl[9] << 32);
            CHECK(fetched == t_fetched, "fetched counter %llu, model %llu", (unsigned long long) fetched, (unsigned long long) t_fetched);
            CHECK(lookups == t_lookups, "lookup counter %llu, model %llu", (unsigned long long) lookups, (unsigned long long) t_lookups);
            CHECK(calls   == t_calls,   "call counter %llu, model %llu",   (unsigned long long) calls,   (unsigned long long) t_calls);
        }

        // coherence: every resident slot equals its page in the host tensors
        std::vector<uint16_t> pk((size_t) N_SLOTS*PAGE*DIM);
        std::vector<uint16_t> pv((size_t) N_SLOTS*PAGE*DIM);
        ggml_backend_tensor_get(pool_k, pk.data(), 0, pk.size()*sizeof(uint16_t));
        ggml_backend_tensor_get(pool_v, pv.data(), 0, pv.size()*sizeof(uint16_t));

        for (int64_t p = 0; p < N_PAGES; ++p) {
            const int32_t s = page_table[p];
            if (s < 0) {
                continue;
            }
            for (int r = 0; r < PAGE; ++r) {
                const int64_t cell = p*PAGE + r;
                if (cell >= N_CELLS) {
                    break;
                }
                const size_t prow = ((size_t) s*PAGE + r)*DIM;
                CHECK(memcmp(&pk[prow], &hk[(size_t) cell*DIM], ROWB) == 0, "K of cell %lld (page %lld, slot %d) differs from the host", (long long) cell, (long long) p, s);
                CHECK(memcmp(&pv[prow], &hv[(size_t) cell*DIM], ROWB) == 0, "V of cell %lld (page %lld, slot %d) differs from the host", (long long) cell, (long long) p, s);
            }
        }

        table_before.assign(page_table, page_table + N_PAGES);
        return true;
    }

    ~kv_test() {
        ggml_backend_buffer_free(buf_host);
        ggml_backend_buffer_free(buf_dev);
        ggml_free(ctx_host);
        ggml_free(ctx_dev);
    }
};

// a selection like a decode step: a recent window, most of the previous call's pages, and some random ones; every
// page contributes most of its cells (selection is in whole blocks, but a page can be named partially)
static std::vector<std::vector<int32_t>> make_lists(std::mt19937 & rng, int64_t head, const std::set<int32_t> & prev, int n_lists, int n_random) {
    std::vector<std::vector<int32_t>> lists;
    for (int l = 0; l < n_lists; ++l) {
        std::set<int32_t> pages;

        for (int64_t c = std::max<int64_t>(0, head - 256); c < head; c += PAGE) {
            pages.insert((int32_t) (c/PAGE));
        }
        // carry over at most ~150 of the previous call's pages per list, so the pages a call names stay well below
        // the pool size (3 lists x (64 recent + 150 carried + 90 random) < 1024 slots)
        const double keep = std::min(0.7, 150.0/(double) std::max<size_t>(prev.size(), 1));
        for (int32_t p : prev) {
            if ((double) (rng() % 1000)/1000.0 < keep) {
                pages.insert(p);
            }
        }
        for (int i = 0; i < n_random; ++i) {
            pages.insert((int32_t) (rng() % N_PAGES));
        }

        std::vector<int32_t> cells;
        for (int32_t p : pages) {
            bool any = false;
            for (int r = 0; r < PAGE; ++r) {
                const int64_t c = (int64_t) p*PAGE + r;
                if (c >= N_CELLS) {
                    break;
                }
                if (rng() % 10 < 8 || (r == PAGE - 1 && !any)) {
                    cells.push_back((int32_t) c);
                    any = true;
                }
            }
        }
        if ((int) cells.size() > STRIDE) {
            cells.resize(STRIDE);
        }
        lists.push_back(std::move(cells));
    }
    return lists;
}

static std::set<int32_t> pages_of(const std::vector<std::vector<int32_t>> & lists) {
    std::set<int32_t> s;
    for (const auto & l : lists) {
        for (int32_t c : l) {
            s.insert(c/PAGE);
        }
    }
    return s;
}

static bool run_tests(kv_test & t) {
    std::mt19937 rng(42);

    // 1. a long run of decode-like steps: rows appended, locality-heavy selections, plus a churn phase with mostly
    //    random pages so the pool fills and evicts
    int64_t head = 512;
    std::set<int32_t> prev;

    for (int call = 0; call < 400; ++call) {
        const bool churn = call >= 250;

        std::vector<int32_t> idxs;
        const int n_new = 1 + (int) (rng() % 4);
        for (int i = 0; i < n_new; ++i) {
            const int64_t cell = head % N_CELLS;
            t.write_row(cell, rng);
            idxs.push_back((int32_t) cell);
            ++head;
        }

        const int n_lists  = 1 + (int) (rng() % 3);
        const int n_random = churn ? 90 : 12;
        const auto lists = make_lists(rng, head % N_CELLS, prev, n_lists, n_random);

        if (!t.step(lists, idxs, false)) {
            fprintf(stderr, "  in call %d of the long run\n", call);
            return false;
        }
        prev = pages_of(lists);
    }
    printf("long run: %llu calls, %llu page lookups, %llu pages fetched (%.1f%% misses)\n",
            (unsigned long long) t.t_calls, (unsigned long long) t.t_lookups, (unsigned long long) t.t_fetched,
            t.t_lookups ? 100.0*t.t_fetched/t.t_lookups : 0.0);

    // 2. a call that selects nothing changes nothing
    {
        const std::vector<int32_t> table = t.table_before;
        const int epoch = t.epoch;
        CHECK(t.step({}, {}, false), "an empty call");
        CHECK(t.table_before == table && t.epoch == epoch, "an empty call changed the state");
    }

    // 3. rows appended to cells of resident pages, with nothing selected, must still reach the pool
    {
        std::vector<int32_t> idxs;
        for (int32_t p = 0; p < (int32_t) N_PAGES && (int) idxs.size() < MAXI; ++p) {
            if (t.table_before[(size_t) p] >= 0) {
                const int64_t cell = (int64_t) p*PAGE + (int) (rng() % PAGE);
                if (cell < N_CELLS) {
                    t.write_row(cell, rng);
                    idxs.push_back((int32_t) cell);
                }
            }
        }
        CHECK(!idxs.empty(), "no resident page to append to");
        CHECK(t.step({}, idxs, false), "append to resident pages"); // verify() compares every resident slot with the host
    }

    // 4. more distinct pages than the pool holds: the flag is set, the pool is full and consistent, nothing is half done
    {
        t.reset_state();

        std::vector<int32_t> cells;
        for (int32_t p = 0; p < 1100; ++p) {
            for (int r = 0; r < PAGE; ++r) {
                cells.push_back(p*PAGE + r);
            }
        }
        CHECK(t.step({ cells }, {}, true), "overflow");
    }

    // 5. and a fresh pool works normally after that
    {
        t.reset_state();
        std::set<int32_t> none;
        for (int call = 0; call < 20; ++call) {
            const auto lists = make_lists(rng, 4000 + 40*call, call ? prev : none, 2, 20);
            CHECK(t.step(lists, {}, false), "a normal call after a reset");
            prev = pages_of(lists);
        }
    }

    return true;
}

int main() {
    ggml_backend_dev_t dev = nullptr;
    kv_stream_run_fn_t run = nullptr;
    kv_host_buft_fn_t  host_buft_fn = nullptr;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
        run          = (kv_stream_run_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_run");
        host_buft_fn = (kv_host_buft_fn_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_host_buffer_type");
        if (run && host_buft_fn) {
            dev = d;
            break;
        }
    }

    if (!dev) {
        printf("no backend with a KV-stream page pool: skipped\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "failed to initialize %s\n", ggml_backend_dev_name(dev));
        return 1;
    }

    int ret = 0;
    {
        kv_test t;
        t.backend = backend;
        t.run     = run;

        if (!t.init(host_buft_fn())) {
            ret = 1;
        } else if (!run_tests(t)) {
            ret = 1;
        } else {
            printf("OK (%s)\n", ggml_backend_dev_name(dev));
        }
    }

    ggml_backend_free(backend);
    return ret;
}
