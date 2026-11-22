#pragma once

#include <stddef.h>
#include <stdint.h>

// Structural ceiling on how many (token, selected-expert) pairs a single
// cache-eligible MUL_MAT_ID node may carry. Shared between the CUDA
// provider (moe-cache.cu, which sizes its per-node pin array and scratch
// reservation off this) and the CPU dispatch path (ggml-cpu.c, which sizes
// its stack-local id/row arrays off the same value) so the two halves of
// this handshake cannot drift apart - they used to be two separately
// maintained "64"s that only a comment tied together. 4096 covers a full
// n_ubatch=512 prefill chunk at up to 8 selected experts/token (the common
// range for today's MoE models); raise further only after checking the
// per-device scratch reservation this adds (grows linearly with this
// value, a few tens of MB at 4096 for typical hidden/ffn sizes).
#define GGML_MOE_CACHE_MAX_BATCH_ROWS 4096

#ifdef __cplusplus
extern "C" {
#endif

// Aggregate cache health, summed across every currently-live session's
// devices - the same numbers moe-cache.cu already logs per device
// (MOE_CACHE_LOG), exposed for a UI/monitoring caller instead of only the
// log. All-zero when no session has recorded anything yet.
// One co-activation edge, real (tensor, expert) identity on both ends -
// see ggml_moe_cache_api::get_co_activation's doc comment for what
// cross_layer means and why this replaced an earlier, opaque-hash version.
struct ggml_moe_cache_co_activation_entry {
	const void * tensor_from;
	int32_t      expert_from;
	const void * tensor_to;
	int32_t      expert_to;
	uint32_t     count;
};

// Router-rank resolution for the residency histogram below. Ranks at or
// beyond this are folded into the last bucket; real top-k is far smaller
// (8 on the models this was built for), so the tail is normally empty.
#define GGML_MOE_CACHE_MAX_RANK 16

struct ggml_moe_cache_summary {
	long long hits;
	long long misses;
	long long evictions;
	long long fill_failures;
	long long admission_skips;
	long long prefetches;
	size_t slots_used;
	size_t slots_total;
	size_t protected_slots;
	double avg_heat;
	size_t allocated_bytes;
	size_t budget_bytes;

	// Step 0 of Atlas-driven cache warming (see ggml_moe_cache.set_atlas) -
	// the live, decaying request-direction centroid, in the same (x,y)
	// affinity space the atlas cells use. req_dir_valid is false until at
	// least one atlas-covered expert has actually been selected; x/y are
	// 0 until then. When multiple devices are live, this is simply
	// whichever device's session was iterated last - fine for the
	// visualization this feeds, not meant to be a precise cross-device
	// aggregate the way the counters above are.
	// Hit/miss split by the router's own rank for that pick (0 = the
	// router's top choice). Answers whether cache misses are concentrated
	// in the low-confidence tail of top-k - which is what decides whether
	// re-ranking toward resident experts could recover them cheaply, or
	// would have to overrule the router's strongest picks.
	// Misses served by running a resident stand-in instead of falling back
	// to CPU compute (GGML_CUDA_MOE_CACHE_SUBSTITUTE). Not folded into hits:
	// a substitution is a miss that was made cheap, not a hit.
	long long substitutions;
	long long substitute_declined;
	// Misses already counted above that turned into a real resident row
	// before plan() returned, via GGML_CUDA_MOE_CACHE_SYNC_FETCH's bounded
	// wait for the fetch just queued for them (off by default).
	long long sync_fetch_hits;

	long long rank_hits[GGML_MOE_CACHE_MAX_RANK];
	long long rank_misses[GGML_MOE_CACHE_MAX_RANK];

	float req_dir_x;
	float req_dir_y;
	int   req_dir_valid;
};

struct ggml_moe_cache_api {
    const void * owner;

    // The scheduler owns one cache session. backends contains the scheduler's
    // actual backend set, so the provider can use only selected CUDA devices.
    void * (*session_create)(void * const * backends, int n_backends);

    // Take an additional reference to an existing session, so two schedulers can
    // share one expert cache. Returns the same pointer; every holder calls
    // session_destroy and only the last release tears down.
    //
    // Used for speculative decoding: the draft head and its target otherwise get
    // a session each, which means two pools, two budgets - split by whichever
    // context initialised first rather than by anything measured - and no shared
    // ranking between the draft's experts and the target's. The draft runs on
    // every decode step, so under one session its experts earn residency on
    // measured access like any other.
    void * (*session_share)(void * session);
    void   (*session_destroy)(void * session);
    // NULL and dormant sessions still create a suppressing thread-local scope.
    void   (*session_enter)(void * session);
    void   (*session_leave)(void * session);
    // As session_enter, but tags the scope with what is computing in it:
    //   bit 0 - serve exact experts only, no stand-ins
    //   bit 1 - this is a speculative draft
    // The draft bit is independent of the exact bit: a draft keeps its own pools,
    // its own heat and its own knob values (a knob may be given a "<NAME>_DRAFT"
    // override) whether or not stand-ins are allowed in it - the same rules as the
    // target, applied within itself. Leave with session_leave as usual.
    void   (*session_enter_scope)(void * session, int flags);

    // Set a policy knob at runtime, overriding the environment; an empty or NULL
    // value clears the override. Only knobs that do not decide an allocation can
    // change this way - expert-cache size, placement and ubatch are fixed once the
    // pools exist. Lets a calibration sweep measure many values in one process
    // instead of reloading the model for each (~80% of a candidate's cost).
    void   (*set_tunable)(const char * name, const char * value);

    // Begin one CPU MUL_MAT_ID node. Returns an opaque plan, or NULL when the
    // stock CPU path should handle the complete node.
    void * (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                    int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens);

    // Mark cache hits and enqueue bounded demand fills for misses. A nonnegative
    // slot index means that the row may be omitted from CPU work only if
    // dispatch subsequently succeeds.
    int (*plan)(void * node, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // Optional, off by default (NULL when the predictor is disabled - the
    // caller must null-check before calling, same convention as every other
    // slot here). Free (input, label) training pair for a live linear
    // predictor: `acts` is the row-major hidden-state input this node's
    // router decision was actually computed from (ids IS the label, already
    // known by this point - this trains the predictor for FUTURE dispatches,
    // it cannot change what this node itself does), one row per token,
    // `act_stride` bytes between rows, `hidden_dim` floats per row. Only the
    // first token of the batch is sampled per call - see moe_cache_train's
    // comment for why one example per dispatch is enough and cheap is safer
    // than complete on this specific hot path.
    // Optional, may be NULL. Hand the predictor an architecture-specific
    // feature block to condition on, for the layer that is about to route.
    //
    // The predictor's default input is the hidden state at the MoE op, which
    // is STALE for cross-layer prediction: the next layer's residual has not
    // been added yet, and that staleness is the measured ceiling on router
    // lookahead (59.3% at depth 1). Some architectures carry a component of
    // the next layer's input that is known exactly and early - Qwen3.8-Flash-
    // Next gathers its whole per-layer n-gram embedding (PLE) once, before
    // the layer loop, and injects a slice of it into every layer's residual.
    // That slice is not a guess about the next layer; it is part of the next
    // layer's input, available now.
    //
    // Generic by construction: this file cannot know what the block means,
    // and models that have nothing to offer simply never call it. The block
    // is appended to the predictor's input vector as-is.
    void (*set_aux_features)(const float * feats, int n_feats);

    // Enumerate the budget partitions worth measuring.
    //
    // The cache's consumers - the reduced pool, the draft's experts, the
    // primary pool - all bid for one number, and which splits are even legal
    // is the arbiter's business, not the caller's: the primary is the
    // remainder and has a floor no combination of shares may cross. A caller
    // that hardcoded its own list would be re-deciding the constraint from
    // outside, and would drift from it the moment the floor changed.
    //
    // Fills out_reduced_pct / out_draft_pct with at most `max` feasible
    // partitions (a draft percentage of -1 means "leave the draft on the
    // weight-based default") and returns how many were written. Ordered with
    // the incumbent first, so a caller that runs out of budget partway has
    // still measured the configuration it would otherwise have shipped.
    int (*partition_candidates)(int * out_reduced_pct, int * out_draft_pct, int max);

    void (*train)(void * node, const int32_t * ids, int n_ids_per_token, int n_tokens,
                  const float * acts, size_t act_stride, int64_t hidden_dim);

    // Dispatch all planned hit rows. Returns 1 only after the complete GPU
    // operation has been accepted. On 0, the caller must restore every row to
    // the normal CPU mapping before worker threads start.
    int (*dispatch)(void * node, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                    const int32_t * slot_idx, const float * const * act_rows);

    // Copy GPU results into dst_rows. On 0, the caller must recompute every
    // skipped row on the CPU.
    int (*collect)(void * node, int n_hits, float * const * dst_rows, int64_t n_out);

    // Releases slot pins and all per-node ownership. Must be called exactly
    // once for every non-NULL begin result, on every success or failure path.
    void (*end)(void * node);

    // Host buffer mutation or teardown notification. Sessions cancel or finish
    // any fill that still reads the supplied range before this call returns.
    void (*invalidate)(const void * base, size_t size);

    // Live hint for the default max batch size a cache-eligible MUL_MAT_ID
    // call may have (the provider still clamps to its own structural
    // ceiling). Callers should pass the real max concurrent sequence count
    // (e.g. llama_context's n_seq_max) before the next session_create(), so
    // the cache doesn't silently disable itself once real concurrency
    // exceeds a default sized for a single interactive session. NULL-safe:
    // callers must check for NULL before calling, same as other optional
    // entries here.
    void (*set_max_batch_hint)(int n_seq_max);

    // The loader already decides whether the model exceeds a safe fraction of
    // host RAM (it disables eager mmap prefetch on that basis). The cache used to
    // re-derive the same fact from /proc/meminfo for its own budgets, so two
    // independent estimates of one thing could disagree. The loader publishes its
    // answer here; -1 means "not told", which keeps the old self-derived path.
    void (*set_host_oversubscribed)(int oversubscribed);

    // Live topic position, for the prompt-cache prewarm link (see
    // prewarm_from_topic below). Copies the decaying (x, y) centroid and, if
    // dims/max_dims are non-null, the fuller embedding it was projected from.
    // Returns false if no routing has happened yet on this session (nothing
    // to save). This is a snapshot for the CALLER to persist - it does not
    // change what the running session is doing.
    bool (*get_topic)(float * out_x, float * out_y, float * dims, int max_dims, int * out_n_dims);

    // Restore a topic position saved earlier (see get_topic) and prewarm the
    // experts nearest it, into free VRAM slots only - the same admission
    // primitive and the same never-evict rule the live atlas warm path uses
    // (moe_cache_atlas_admit), just seeded from a caller-supplied position
    // instead of the session's own live req_dir. Meant for a server restoring
    // a long-idle cached prompt: the topic it saved when that prompt went
    // idle is usually still a good prior for what its continuation needs.
    void (*prewarm_from_topic)(float x, float y, int top_k_per_tensor);

    // Aggregate hit/miss counts summed across every currently-live session's
    // devices. Meant for calibration/benchmarking callers that create one
    // context at a time (the common case there), not general production
    // monitoring - with multiple concurrent sessions this sums across all of
    // them, not just "the" one a caller might have in mind. Writes 0/0 if no
    // session has recorded anything yet. NULL-safe like the other optional
    // entries here.
    void (*get_stats)(long long * out_hits, long long * out_misses);

    // Snapshot the live per-(layer,expert) cache state for a debugging/demo
    // UI view (the "Brain" page). One byte per (layer,expert) cell packed
    // as (tier << 6) | heat - see moe_cache_get_expert_map()'s own comment
    // in moe-cache.cu for the exact encoding. out_rows/out_cols are always
    // written (0 on failure). Returns 1 on success, 0 if no session has
    // cached anything yet or out_bytes is too small for the real grid (in
    // which case out_rows/out_cols still report the real shape so the
    // caller can retry with a bigger buffer). NULL-safe like get_stats.
    int (*get_expert_map)(uint8_t * out_bytes, int max_bytes, int * out_rows, int * out_cols);

    // Which (layer,expert) cells served as a resident substitution stand-in
    // SINCE THE LAST CALL - a bitset, 1 bit/cell, same rows/cols shape as
    // get_expert_map. Unlike get_expert_map (a live snapshot, re-readable
    // any time), this is READ-AND-CLEARED: correct for exactly one poller.
    // NULL-safe, same failure/retry convention as get_expert_map (0 return
    // with the real shape in out_rows/out_cols if out_bits is too small).
    int (*get_substitute_map)(uint8_t * out_bits, int max_bytes, int * out_rows, int * out_cols);

    // Which (layer,expert) cells belong to the speculative draft rather than the
    // target - a bitset, same shape as get_expert_map. A live snapshot (role does
    // not change), so unlike get_substitute_map it is not read-and-cleared.
    int (*get_draft_map)(uint8_t * out_bits, int max_bytes, int * out_rows, int * out_cols);

    // Live per-(layer,expert) neuron concentration, for the Brain/Atlas
    // view - a cheap O(n) dead-neuron-fraction proxy (0.0-1.0, share of
    // this expert's neurons contributing under 1% of its own mean value),
    // -1.0 for a cell with no data yet. Same rows/cols shape convention
    // as get_expert_map, writes a float per cell instead of a bit. Only
    // produces real data when GGML_CUDA_MOE_CACHE_NEURON_HEAT is set;
    // NULL-safe, returns 0 (no data) otherwise. Same failure/retry
    // convention as get_expert_map/get_substitute_map for an
    // undersized buffer.
    int (*get_neuron_concentration_map)(float * out_values, int max_floats, int * out_rows, int * out_cols);

    // Debug only, NULL-safe: assert that each compacted hit row is about to be
    // served by a slot actually holding the expert that row asked for. The
    // slot CONTENTS have been verified correct, so a corruption that survives
    // that check has to be correct weights reaching the wrong row - which is
    // exactly what this catches. Enabled by GGML_CUDA_MOE_CACHE_VERIFY_ROWS.
    void (*verify_rows)(void * node, int n_hits, const int32_t * slot_idx,
                        const int32_t * experts);

    // Host-side hot-expert buffer. Returns a pointer to this expert's weights
    // in memory the cache owns, or NULL to use the caller's own (mmap'd)
    // pointer. Called on the CPU expert path for every expert of every MoE
    // node, so it must be cheap and lock-free.
    //
    // The point is enforcement. mmap leaves residency to the kernel, which
    // evicts by recency and cannot tell an expert that fires on most tokens
    // from one never selected - measured, that costs 21x under memory pressure.
    // Weights promoted into this buffer are held in memory we control, and the
    // corresponding mmap pages are released so nothing is resident twice.
    const void * (*host_ptr)(const void * host_base, int expert);

    // Speculative prefetch. `ids` are the experts a *later* layer is predicted
    // to select, so their fills can be issued while the current layer is still
    // computing - converting a blocking transfer into an overlapped one.
    //
    // This is the mechanism the MoE-offload literature converged on
    // (Speculating Experts, PILOT, Fate, FineMoE): prediction is only ~77%
    // accurate, so these fills must never displace an expert the cache is
    // confident about. Speculation loses ties to certainty.
    //
    // `depth` is how many MoE layers ahead this prediction targets (1 =
    // immediate next layer, using the freshest, most accurate router
    // state; higher = farther and measurably less accurate - 59.3% / 48.0%
    // / 41.1% precision at depth 1/2/3). Needed so cross-depth-agreement
    // speculative eviction (GGML_CUDA_MOE_CACHE_SPEC_EVICT_MODE=agree) can
    // require a *closer* depth to confirm a farther one specifically,
    // rather than merely "seen twice regardless of which depths agreed" -
    // the latter also accepts weak depth-3-confirms-depth-2 corroboration
    // and measurably dilutes the effect (see docs/index.html).
    void (*prefetch)(const void * host_base, const int32_t * ids, int n_ids, int depth);

    // Full-layer prefill double buffer, split into a build-time and a
    // compute-time half - CUDA graph capture can only ever be active during
    // compute, so anything that allocates (cudaMalloc) has to happen at
    // build time to be unconditionally capture-safe, while the thing that
    // runs every call (the copy itself) has to happen at compute time to
    // actually refresh the buffer on every graph replay, not just once.
    //
    // Call at graph-BUILD time (once per layer, never at compute time):
    // registers "when host_base's node is dispatched, prefetch
    // next_host_base (its complete byte size, not a per-row scratch size)"
    // and eagerly allocates everything that will need. next_n_expert is the
    // successor tensor's own expert count (its ne[2]) - needed so
    // prefill_advance can split the copy per expert row and skip whichever
    // rows are already resident in the decode-time LFRU pool (a
    // device-to-device copy instead of one over PCIe). Pass 0 if unknown, in
    // which case the whole tensor is always copied over PCIe as one block
    // (the pre-hit-D2D behavior). NULL-safe.
    void (*prefill_register_successor)(
            const void * host_base, const void * next_host_base, size_t next_tensor_bytes,
            int64_t next_n_expert);

    // Call at compute time, from inside the consumer that is dispatching
    // host_base's own node (e.g. ggml_cuda_mul_mat_id) - kicks off the copy
    // for whatever tensor was registered as host_base's successor, on a
    // private prefill stream. origin_stream (a cudaStream_t, opaque here
    // since this header is shared with plain-C callers) is unused by the
    // copy itself but kept in the signature for symmetry with prefill_wait/
    // prefill_release. The copy deliberately is NOT made part of whatever
    // CUDA graph capture may currently be active on origin_stream - see the
    // implementation's own design note for why (capture granularity for this
    // op_offload path is roughly one region per layer, so a cross-layer
    // producer/consumer relationship almost never shares a single capture).
    // Never allocates. NULL-safe.
    void (*prefill_advance)(const void * host_base, void * origin_stream);

    // Blocks the calling thread, if needed, until the copy started by a
    // matching prefill_advance call has actually finished (searched for
    // internally - the caller doesn't need to know which slot it landed in,
    // since it's typically a different call site than the one that issued
    // the prefetch), then returns the device pointer to compute against - or
    // NULL if no live prefetch matches (never started, already released, or
    // a different tensor occupies the slot for this shape), in which case
    // the caller must fall back to its normal path exactly as if prefetch
    // had never been called. A real (host) synchronization rather than a
    // stream-ordered one - see prefill_advance's doc comment for why.
    // consumer_stream (a cudaStream_t, opaque here since this header is
    // shared with plain-C callers) is kept for symmetry with prefill_release,
    // whose consumer_stream is genuinely used. On a non-NULL return,
    // *out_slot receives the slot found - pass it to prefill_release once the
    // caller is done reading. out_slot may be NULL if the caller has no
    // matching prefill_release call (e.g. just probing).
    const void * (*prefill_wait)(const void * host_base, size_t tensor_bytes, void * consumer_stream, int * out_slot);

    // Must be called once the caller has issued (not necessarily completed -
    // this only needs to be ordered on consumer_stream) every read of the
    // buffer prefill_wait returned, passing the slot *out_slot reported, so
    // the slot can be safely overwritten by a future prefetch two calls from
    // now (the ping-pong period). Skipping this call is safe but pessimistic,
    // never incorrect - see the implementation's own comment for why.
    // NULL-safe like the other optional entries here.
    void (*prefill_release)(const void * host_base, size_t tensor_bytes, int slot, void * consumer_stream);

    // Called from ggml_backend_sched_compute_splits' existing "copy only the
    // experts that are used" logic (ggml-backend.cpp), for one selected
    // expert row, before that code falls back to its normal H2D path for it.
    // If this expert is already resident in the decode-time LFRU pool for
    // this device, issues a device-to-device copy of expert_size bytes from
    // the resident slot into dst_ptr, ordered on the same stream backend's
    // own async tensor-copy calls use (backend is a ggml_backend_t, opaque
    // here since this header is shared with plain-C callers and has no
    // ggml-backend.h dependency of its own - the implementation extracts its
    // stream the same way ggml_backend_cuda_set_tensor_async does, from
    // backend->context), and returns true - the caller should then exclude
    // this row from whatever H2D it was about to do. Returns false
    // (host_base/expert not resident, or no moe-cache session live) with no
    // side effects, in which case the caller's normal H2D path is unchanged.
    // This is the scheduler-level integration point for LFRU-aware "skip the
    // PCIe copy for what's already resident" - upgrading the existing
    // sparse-copy mechanism in place, not a parallel path to it. Pins the
    // source slot against eviction until the copy is actually complete (via
    // a stream callback), so it's safe to call this from inside an active
    // split computation. NULL-safe.
    bool (*moe_lfru_copy_expert)(
            const void * host_base, int32_t expert, size_t expert_size, void * dst_ptr, void * backend);

    // Batched form of moe_lfru_copy_expert, for the case ggml-backend.cpp's
    // sparse-copy actually has: many candidate expert ids for the same
    // weight tensor, checked in one pass. Checks residency for every id in
    // ids[0..n_ids), issues a device-to-device copy for each resident one
    // (into dst_base + ids[i]*expert_size) and sets out_hit[i] to 1 - all on
    // the same ordering guarantee as moe_lfru_copy_expert. The difference is
    // the reader-pin release: every hit in this call shares ONE stream
    // callback instead of one each, since a real batch (e.g. 121 experts hit
    // in a single warm-cache prefill, measured) would otherwise pay a
    // cudaLaunchHostFunc dispatch (~30-50us, see FreeToken's cpu_executor.py
    // for the same measurement in a different context) per expert for no
    // reason - the pins all become releasable at the same real time anyway,
    // once this call's D2D copies are done. out_hit must have room for
    // n_ids bytes (left at 0 for misses, the caller's own responsibility to
    // zero first if it matters). Returns the number of hits. NULL-safe.
    int (*moe_lfru_copy_experts)(
            const void * host_base, size_t expert_size, const int32_t * ids, int n_ids,
            void * dst_base, uint8_t * out_hit, void * backend);

    // Real-fetch pacing signal for the dynamic substitution gate (see
    // moe_cache_substitute_min_rank in moe-cache.cu): call this once, right
    // after issuing a genuine H2D fetch (a cache miss that could not be
    // served D2D), passing the backend the copy was issued on. Enqueues a
    // stream callback - same cudaLaunchHostFunc mechanism the reader-release
    // pins already use - that fires once every op queued before it,
    // including the fetch just issued, has actually finished on the GPU.
    // The callback measures real wall-clock elapsed time (issue to genuine
    // completion, not merely to enqueue) and folds it into the device's
    // pacing EMA, which is what decides how far the substitution gate opens.
    // Deliberately does not block the caller - the whole point is to learn
    // how backed up the fetch pipeline really is without adding a wait of
    // its own. NULL-safe; a no-op if no CUDA moe-cache session exists.
    void (*track_fetch_pace)(void * backend);

    // Aggregate cache health summary - see ggml_moe_cache_summary. Always
    // writes *out (zeroed on failure). NULL-safe like the other optional
    // entries here.
    void (*get_summary)(struct ggml_moe_cache_summary * out);

    // Step 0 of Atlas-driven cache warming: registers each expert's measured
    // topic-affinity position (see llama-expert-atlas / docs/index.html's
    // Brain-Atlas section) so moe-cache can track, live, which direction in
    // that same space the current request is trending toward - a decaying
    // centroid of already-selected experts' positions, updated on real
    // demand hits. Purely observational for now: this only maintains and
    // logs the signal, nothing reads it to promote or evict anything yet.
    // That's deliberate - the same "prove the signal is real before acting
    // on it" discipline every other predictive mechanism in this cache
    // followed (router-lookahead, speculative eviction).
    //
    // Category names/angles are NOT passed here on purpose: the warming
    // score this will eventually feed is a plain cosine similarity in (x,y)
    // space between a candidate expert and the live request centroid -
    // never needs to name which category that direction corresponds to.
    // Keeping labels out of this low-level, dependency-light module also
    // avoids pulling a JSON parser into ggml-cuda; the atlas file itself is
    // parsed once at the common/ layer, which already has one.
    //
    // host_base identifies which tensor's experts these entries belong to,
    // matching moe_cache_key's own convention. Safe to call once per
    // (host_base) with that tensor's full set of measured cells; a second
    // call for a host_base already registered replaces its entries. NULL-
    // safe. n_cells may be 0 (registers nothing, still valid).
    // dims/n_dims carry the FULL discovered embedding each (x,y) was
    // projected from - n_dims floats per cell, row-major, n_cells rows.
    // Optional: dims may be NULL / n_dims 0, in which case scoring falls back
    // to the 2D position. Passing it matters because the projection is lossy
    // to the point of being useless as a signal: measured on gemma-4, the
    // correlation between 2D atlas distance and distance in the 28-dim
    // embedding it came from is -0.003, i.e. noise, while the embedding
    // itself is what the eviction and warming decisions are meant to consult.
    void (*set_atlas)(
            const void * host_base, const int32_t * expert,
            const float * x, const float * y, const float * spec,
            const float * dims, int n_dims, int n_cells);

    // Track 1 steps 4a/4b's co-activation data, exported as real (tensor,
    // expert) identity - not the opaque hash the first version of this
    // mistakenly stored, which turned out to have thrown away the one
    // thing worth keeping (see moe_cache_edge's doc comment in moe-cache.cu
    // for the full story). cross_layer selects which counter: 0 = 4a
    // (within one routing decision, undirected), 1 = 4b (layer L's top
    // pick -> layer L+1's, directed). Writes up to max_entries of the
    // highest-count edges into out (already sorted, highest count first)
    // and returns how many were written - a top-K query, not a full dump,
    // since a caller like a visualization only ever wants the significant
    // edges. NULL-safe; returns 0 if no CUDA moe-cache session exists yet.
    int (*get_co_activation)(
            int cross_layer, struct ggml_moe_cache_co_activation_entry * out, int max_entries);
};

extern struct ggml_moe_cache_api ggml_moe_cache;
void ggml_moe_cache_unregister(const void * owner);

#ifdef __cplusplus
}
#endif
