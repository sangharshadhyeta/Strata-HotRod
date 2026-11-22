#include "kv-stream.cuh"

// See kv-stream.cuh and docs/kv-streaming-s3d-design.md. The CLOCK page cache follows the design of Strata's KV
// streaming (MIT licensed, https://github.com/Niko1221/Strata); the code here is written against this fork's layout.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#define KV_STREAM_THREADS 1024 // the resolve block; one sweep step looks at this many consecutive slots
#define KV_STREAM_MIN_SLOTS 1024

struct kv_stream_view {
    int32_t * page_table;
    int32_t * slot_page;
    int32_t * slot_stamp;
    int32_t * slot_ref;
    int32_t * miss_page;
    int32_t * miss_slot;
    int32_t * ctl;
};

static __device__ __forceinline__ kv_stream_view kv_stream_make_view(int32_t * state, int64_t n_pages, int64_t n_slots) {
    kv_stream_view v;
    v.page_table = state;
    v.slot_page  = v.page_table + n_pages;
    v.slot_stamp = v.slot_page  + n_slots;
    v.slot_ref   = v.slot_stamp + n_slots;
    v.miss_page  = v.slot_ref   + n_slots;
    v.miss_slot  = v.miss_page  + n_slots;
    v.ctl        = v.miss_slot  + n_slots;
    return v;
}

// Copy nbytes (a multiple of 16) between 16-byte aligned addresses with the threads of a block.
static __device__ __forceinline__ void kv_stream_copy_bytes(char * dst, const char * src, size_t nbytes) {
    int4 *       d = (int4 *) dst;
    const int4 * s = (const int4 *) src;
    const size_t n = nbytes / 16;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        d[i] = s[i];
    }
}

// A 64-bit counter stored as two int32 (low word first): the state blob only guarantees 4-byte alignment.
static __device__ __forceinline__ void kv_stream_add_u64(int32_t * p, unsigned int v) {
    const unsigned int lo  = (unsigned int) p[0];
    unsigned int       hi  = (unsigned int) p[1];
    const unsigned int sum = lo + v;
    if (sum < lo) {
        hi++;
    }
    p[0] = (int32_t) sum;
    p[1] = (int32_t) hi;
}

// Exclusive prefix sum of one int per thread over a block of KV_STREAM_THREADS threads (32 warps);
// `total` is the sum over the block. Contains __syncthreads: every thread of the block must call it.
static __device__ int kv_stream_block_scan(int v, int * warp_sums, int & total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    int x = v; // inclusive scan inside the warp
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xFFFFFFFF, x, o);
        if (lane >= o) {
            x += y;
        }
    }

    if (lane == 31) {
        warp_sums[warp] = x;
    }
    __syncthreads();

    if (warp == 0) { // the first warp scans the 32 warp sums
        int t = warp_sums[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFF, t, o);
            if (lane >= o) {
                t += y;
            }
        }
        warp_sums[lane] = t;
    }
    __syncthreads();

    total = warp_sums[31];
    const int excl = x - v + (warp > 0 ? warp_sums[warp - 1] : 0);
    __syncthreads();

    return excl;
}

// ---------------------------------------------------------------------------------------------------------------------
// append: a cell was just written to the host tensors; if its page is resident, refresh the pool row.

static __global__ void kv_stream_append_kernel(
        int32_t * state, int64_t n_pages, int64_t n_slots, int page, int64_t n_cells,
        const char * host_k, const char * host_v, char * pool_k, char * pool_v,
        size_t row_bytes_k, size_t row_bytes_v, const void * idxs, int idxs_i64, int n_idxs) {
    const kv_stream_view sv = kv_stream_make_view(state, n_pages, n_slots);

    for (int i = blockIdx.x; i < n_idxs; i += gridDim.x) {
        const int64_t cell64 = idxs_i64 ? ((const int64_t *) idxs)[i] : (int64_t) ((const int32_t *) idxs)[i];
        if (cell64 < 0 || cell64 >= n_cells) {
            continue;
        }
        const int cell = (int) cell64;

        const int slot = sv.page_table[cell / page];
        if (slot < 0) {
            continue; // not resident (resolve copies the whole page, including this row)
        }

        const int64_t row = (int64_t) slot*page + cell % page;

        kv_stream_copy_bytes(pool_k + row*row_bytes_k, host_k + (int64_t) cell*row_bytes_k, row_bytes_k);
        kv_stream_copy_bytes(pool_v + row*row_bytes_v, host_v + (int64_t) cell*row_bytes_v, row_bytes_v);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// resolve: one block. 1. look every selected page up, 2. pick a victim slot for each miss, 3. book-keeping.
// The pages are copied by kv_stream_copy_kernel afterwards.

static __global__ void __launch_bounds__(KV_STREAM_THREADS) kv_stream_resolve_kernel(
        int32_t * state, int64_t n_pages, int64_t n_slots, int page,
        const int32_t * lists, const int32_t * counts, int n_lists, int list_stride) {
    __shared__ int s_nmiss;     // pages claimed this call (can exceed n_slots only when the caller broke the capacity rule)
    __shared__ int s_nhit;      // distinct resident pages this call named
    __shared__ int s_overflow;
    __shared__ int warp_sums[32];

    const kv_stream_view sv = kv_stream_make_view(state, n_pages, n_slots);

    const int epoch = sv.ctl[0] + 1;

    if (threadIdx.x == 0) {
        s_nmiss    = 0;
        s_nhit     = 0;
        s_overflow = 0;
    }
    __syncthreads();

    // 1. lookups. A resident page is stamped with this epoch (so the sweep below never takes its slot) and gets its
    //    reference bit. A missing page is claimed exactly once (-1 -> -2) and appended to the miss list.
    for (int l = 0; l < n_lists; ++l) {
        const int32_t * list = lists + (int64_t) l*list_stride;
        const int       n    = counts[l];

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int cell = list[i];
            if (cell < 0) {
                continue;
            }

            const int pg = cell / page;
            if (pg >= n_pages) {
                continue;
            }

            const int old = atomicCAS(&sv.page_table[pg], -1, -2);
            if (old >= 0) {
                if (atomicExch(&sv.slot_stamp[old], epoch) != epoch) {
                    atomicAdd(&s_nhit, 1);
                }
                sv.slot_ref[old] = 1;
            } else if (old == -1) {
                const int m = atomicAdd(&s_nmiss, 1);
                if (m < n_slots) {
                    sv.miss_page[m] = pg;
                } else {
                    sv.page_table[pg] = -1; // no room in the miss list: give the claim back
                    s_overflow = 1;
                }
            }
            // old == -2: another thread of this call claimed it already
        }
    }
    __syncthreads();

    const int n_miss = min(s_nmiss, (int) n_slots);

    // 2. victims: a CLOCK sweep (second chance). One step looks at KV_STREAM_THREADS consecutive slots starting at the
    //    hand, one slot per thread (n_slots >= KV_STREAM_THREADS, so no slot is looked at twice in a step).
    //    A slot used by this call is skipped; a slot with its reference bit set loses the bit; any other is a candidate.
    //    The candidates of a step are numbered in thread order and the first ones, as many as are still needed, are taken.
    int assigned = 0;
    int hand     = sv.ctl[1];

    const int max_steps = 2*(int) ((n_slots + KV_STREAM_THREADS - 1)/KV_STREAM_THREADS) + 2; // two rotations clear every reference bit

    for (int step = 0; step < max_steps && assigned < n_miss; ++step) {
        const int slot = (int) ((hand + threadIdx.x) % n_slots);

        bool candidate = false;
        if (sv.slot_stamp[slot] != epoch) {
            if (sv.slot_ref[slot] != 0) {
                sv.slot_ref[slot] = 0;
            } else {
                candidate = true;
            }
        }

        int total;
        const int rank = kv_stream_block_scan(candidate ? 1 : 0, warp_sums, total);
        const int want = n_miss - assigned;

        if (candidate && rank < want) {
            const int m        = assigned + rank;
            const int old_page = sv.slot_page[slot];
            const int pg       = sv.miss_page[m];

            if (old_page >= 0) {
                sv.page_table[old_page] = -1;
            }
            sv.slot_page[slot]  = pg;
            sv.page_table[pg]   = slot;
            sv.slot_stamp[slot] = epoch;
            sv.slot_ref[slot]   = 1;
            sv.miss_slot[m]     = slot;
        }

        assigned += min(total, want);
        hand = (int) ((hand + KV_STREAM_THREADS) % n_slots);
        __syncthreads();
    }

    // pages that found no slot (more distinct pages than free slots: the caller broke the capacity rule) go back to
    // "not resident"
    for (int m = assigned + threadIdx.x; m < n_miss; m += blockDim.x) {
        sv.page_table[sv.miss_page[m]] = -1;
    }
    __syncthreads();

    // 3. book-keeping
    if (threadIdx.x == 0) {
        sv.ctl[0] = epoch;
        sv.ctl[1] = hand;
        sv.ctl[2] = assigned;
        if (assigned < n_miss || s_nmiss > n_slots || s_overflow != 0) {
            sv.ctl[3] = 1;
        }
        kv_stream_add_u64(sv.ctl + 4, (unsigned int) assigned);
        kv_stream_add_u64(sv.ctl + 6, (unsigned int) (s_nhit + n_miss));
        kv_stream_add_u64(sv.ctl + 8, 1u);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// copy: the pages resolve just assigned, from the host tensors into their slots.

static __global__ void kv_stream_copy_kernel(
        int32_t * state, int64_t n_pages, int64_t n_slots, int page, int64_t n_cells,
        const char * host_k, const char * host_v, char * pool_k, char * pool_v, size_t row_bytes_k, size_t row_bytes_v) {
    const kv_stream_view sv = kv_stream_make_view(state, n_pages, n_slots);

    const int n_miss = sv.ctl[2];

    for (int m = blockIdx.x; m < n_miss; m += gridDim.x) {
        const int     pg   = sv.miss_page[m];
        const int     slot = sv.miss_slot[m];
        const int64_t c0   = (int64_t) pg*page;
        // the last page can be cut short by the end of the cache
        const int64_t rows = (n_cells - c0 < (int64_t) page) ? (n_cells - c0) : (int64_t) page;

        if (rows <= 0) {
            continue;
        }

        kv_stream_copy_bytes(pool_k + (int64_t) slot*page*row_bytes_k, host_k + c0*row_bytes_k, (size_t) rows*row_bytes_k);
        kv_stream_copy_bytes(pool_v + (int64_t) slot*page*row_bytes_v, host_v + c0*row_bytes_v, (size_t) rows*row_bytes_v);
    }
}

// ---------------------------------------------------------------------------------------------------------------------

static void kv_stream_check(const ggml_cuda_kv_stream & s) {
    GGML_ASSERT(s.state != nullptr && s.pool_k != nullptr && s.pool_v != nullptr);
    GGML_ASSERT(s.host_k != nullptr && s.host_v != nullptr);
    GGML_ASSERT(s.page > 0 && s.n_pages > 0 && s.n_cells > 0);
    GGML_ASSERT(s.n_slots >= KV_STREAM_MIN_SLOTS && "the resolve block needs at least 1024 slots");
    GGML_ASSERT(s.row_bytes_k % 16 == 0 && s.row_bytes_v % 16 == 0);
}

void ggml_cuda_kv_stream_append(const ggml_cuda_kv_stream & s, const void * idxs, bool idxs_i64, int n_idxs, cudaStream_t stream) {
    if (n_idxs <= 0) {
        return;
    }

    kv_stream_check(s);

    const int n_blocks = n_idxs < 128 ? n_idxs : 128;

    kv_stream_append_kernel<<<n_blocks, 128, 0, stream>>>(
            s.state, s.n_pages, s.n_slots, s.page, s.n_cells,
            s.host_k, s.host_v, s.pool_k, s.pool_v, s.row_bytes_k, s.row_bytes_v, idxs, idxs_i64 ? 1 : 0, n_idxs);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_kv_stream_resolve(const ggml_cuda_kv_stream & s, const int32_t * lists, const int32_t * counts,
        int n_lists, int list_stride, cudaStream_t stream) {
    if (n_lists <= 0) {
        return; // nothing selected: state and counters stay as they are
    }

    kv_stream_check(s);

    kv_stream_resolve_kernel<<<1, KV_STREAM_THREADS, 0, stream>>>(
            s.state, s.n_pages, s.n_slots, s.page, lists, counts, n_lists, list_stride);
    CUDA_CHECK(cudaGetLastError());

    kv_stream_copy_kernel<<<96, 128, 0, stream>>>(
            s.state, s.n_pages, s.n_slots, s.page, s.n_cells,
            s.host_k, s.host_v, s.pool_k, s.pool_v, s.row_bytes_k, s.row_bytes_v);
    CUDA_CHECK(cudaGetLastError());
}

#else // GGML_USE_HIP || GGML_USE_MUSA

void ggml_cuda_kv_stream_append(const ggml_cuda_kv_stream & s, const void * idxs, bool idxs_i64, int n_idxs, cudaStream_t stream) {
    GGML_UNUSED_VARS(s, idxs, idxs_i64, n_idxs, stream);
    GGML_ABORT("KV streaming is only supported on NVIDIA CUDA");
}

void ggml_cuda_kv_stream_resolve(const ggml_cuda_kv_stream & s, const int32_t * lists, const int32_t * counts,
        int n_lists, int list_stride, cudaStream_t stream) {
    GGML_UNUSED_VARS(s, lists, counts, n_lists, list_stride, stream);
    GGML_ABORT("KV streaming is only supported on NVIDIA CUDA");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
