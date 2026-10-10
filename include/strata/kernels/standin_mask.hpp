// include/strata/kernels/standin_mask.hpp - the stand-in for every miss, as a mask on the router (SPEC.md 13.20, step 2).
//
// A routed expert that is not in VRAM is not fetched: the router is restricted to the experts that are, so every pick is
// computed on the GPU. The router's top-k renormalises its weights over the chosen k, so the softmax denominator cancels:
// setting the logits of non-resident experts to minus infinity before the top-k gives exactly "the k best resident experts,
// renormalised" (the decided stand-in rule, with the stand-in taking its own router score).
//
// Off unless STRATA_STANDIN=1. Decode only: a window of more than `standin_max_rows()` rows is a prompt chunk and is routed
// exactly. A layer with fewer than k resident experts is left to the unmasked router (the mask never leaves fewer than k).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The device residency table (layer * n_expert + expert -> slot, or negative when not in VRAM) of the single GPU, set once
/// when the table is staged; nullptr (the default) disables the mask. The pointer stays valid for the run: the table is
/// refreshed in place after each swap round.
void standin_set_table(const int32_t* d_res, int n_expert);

/// STRATA_STANDIN=1 and a table is set.
bool standin_enabled();

/// Rows above this are prompt chunks (never masked).
int standin_max_rows();

/// Masks `n_rows` rows of `logits` (n_expert floats each, device) of `layer`; no-op when disabled or `n_rows` is too large.
void standin_mask(float* logits, int n_rows, int n_expert, int k, int64_t layer, void* stream);

}  // namespace strata::kernels
