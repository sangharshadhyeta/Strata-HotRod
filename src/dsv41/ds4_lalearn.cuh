// src/dsv41/ds4_lalearn.cuh - the DeepSeek engine's learned look-ahead (STRATA_LA_LEARN), ported from Strata's Qwen engine
// (src/kernels/cuda/native_router.cu, la_learn_*).  For each source layer s and distance d (kDist) the read-ahead's guess of
// layer s+d's experts is that layer's router applied to x_s (the plain guess) plus a learned low-rank correction B (A x_s).
// The leader's slot (plain or one of three corrections) is what the look-ahead publishes; the training is one batch per decode
// token, on a side stream.  Single GPU: every state lives on the device of the trunk's first layer.
#pragma once

#include <cuda_runtime.h>

#include <string>

namespace strata::ds4::la {

constexpr int kDist[] = {1, 2, 4};   // the look-ahead distances: 1 and 2 are published by the engine, 4 is trained and reported only
constexpr int kNd = 3;               // entries in kDist

// Allocates the learner for n_layers trunk layers of n_embd inputs and n_exp routed experts each (n_exp <= 512; the rows are padded
// to 512 with experts that are never scored).  lo: the true top-k's floor (a true pick below it is scored).  The current device.
bool alloc(int n_layers, int n_embd, int n_exp, int lo, std::string& msg);
bool active();
// Frees the learner's device memory, streams and events (the file is not written: save first).
void release();
// The device pointers of each trunk layer's resident-expert table (vram_tab, n_exp entries; nonzero = resident), one per layer,
// uploaded once.  The learner's resident flags are refreshed from them at each window's end.
bool set_resident_tables(const unsigned long long* const* host_ptrs, int L);

// The forward of source layer s at distance dist, on cs (n rows): x = the source's MoE input (H per row), gl = the plain logits of
// layer s+dist on x (row stride 1024, n_exp valid).  Writes the leader's logits back into gl (the first n_exp entries of each row).
void forward(int s, int dist, const float* x, float* gl, int n, cudaStream_t cs);
// Layer l's router logits for this window (row stride 1024, n_exp valid), copied on cs; the padding is kept at -1e30.
void label(int l, const float* glog, int n, cudaStream_t cs);
// Before a decode window's first layer is enqueued on cs: waits for the previous window's batch.
void pre_launch(cudaStream_t cs);
// After a decode window's last layer is enqueued on cs: the batch on the side stream (after the window's end).
void post_launch(cudaStream_t cs, int n);

// The per-(layer, dist) state: the leader slot, the picks and each slot's all-time hits (slot 0 is the plain router).
// Waits for the last batch.  False when the learner is off or dist is not in kDist.
bool counters(int l, int dist, int& leader, unsigned long long& picks, unsigned long long hits[4]);
// The learner's report lines (leader hit rates, the chosen ranks, the resets); empty when off.
std::string report();

// The file: the header (magic, version, layers, n_embd, n_exp, the distances and the correction ranks), then per (layer, dist) one
// word (the leader's rank, | 256 when frozen), then the leader's A and B as fp16 for each (layer, dist) with a correction.
// save writes to a temporary file and renames it.  load refuses a file whose header differs from this learner's.
bool save(const char* path, std::string& msg);
bool load(const char* path, std::string& msg);

}  // namespace strata::ds4::la
