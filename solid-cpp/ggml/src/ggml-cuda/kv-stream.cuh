#pragma once

#include "common.cuh"

// KV streaming, step S3d-2: the device side of the resident page pool of one streamed layer
// (docs/kv-streaming-s3d-design.md, sections 3, 4 and 6).
//
// The K/V rows live in two host tensors (pinned host memory the device reads in place). A pool in device memory holds
// `n_slots` pages of `page` consecutive cells; `state` is the I32 blob whose layout is given by
// ggml_kv_stream_state_ints() in ggml.h:
//   page_table[n_pages]   page -> slot, -1 not resident, -2 claimed by the call in progress (never visible between calls)
//   slot_page[n_slots]    slot -> page, -1 free
//   slot_stamp[n_slots]   the call (epoch) that last used the slot
//   slot_ref[n_slots]     CLOCK reference bit
//   miss_page[n_slots]    pages this call had to fetch
//   miss_slot[n_slots]    the slot each of them went to
//   ctl[16]               [0] epoch  [1] clock hand  [2] pages fetched by the last call  [3] overflow flag
//                         [4..5] fetched pages, [6..7] page lookups, [8..9] calls (each a 64-bit value, low word first)
//
// Invariant between calls: the host tensors are always right, and a resident slot equals its page in the host tensors.

struct ggml_cuda_kv_stream {
    int32_t *    state;        // device memory
    int64_t      n_pages;      // pages the host tensors can address
    int64_t      n_slots;      // pages the pool holds; at least 1024
    int          page;         // cells per page
    int64_t      n_cells;      // cells in the host tensors
    const char * host_k;       // host K: n_cells rows of row_bytes_k
    const char * host_v;       // host V: n_cells rows of row_bytes_v
    char *       pool_k;       // device: n_slots*page rows of row_bytes_k
    char *       pool_v;       // device: n_slots*page rows of row_bytes_v
    size_t       row_bytes_k;  // a multiple of 16
    size_t       row_bytes_v;  // a multiple of 16
};

// Keep resident pages coherent with cells that were just written to the host tensors: for each cell of idxs (device
// memory, int32 or int64 (the cache's k_idxs is int64)), if its page is resident, copy its K and V row from the host
// tensors into the pool. A page that is not resident needs nothing: resolve copies it whole, including the new rows.
// One stream only: a cell id is an index into the host tensors.
void ggml_cuda_kv_stream_append(const ggml_cuda_kv_stream & s, const void * idxs, bool idxs_i64, int n_idxs, cudaStream_t stream);

// Make every page named by the lists resident. `lists` holds n_lists lists of cell indices, list l at
// lists + l*list_stride, with counts[l] valid entries (the layout flash_attn_mask_to_sparse_indices writes); both are
// device memory. A resident page is marked used, a missing page gets a victim slot by a CLOCK sweep that never takes a
// slot used by this call, and is copied in from the host tensors. The distinct pages of one call must fit in the pool;
// if they do not, the pages that found no slot are left not resident and ctl[3] is set.
void ggml_cuda_kv_stream_resolve(const ggml_cuda_kv_stream & s, const int32_t * lists, const int32_t * counts,
        int n_lists, int list_stride, cudaStream_t stream);
