# Edge0-145B expert predictor study (15 Nov 2026)

Trace: `STRATA_ROUTER_TRACE=<file>` with the stand-ins and the learner off (`STRATA_ROUTE_RESIDENT=0`, `STRATA_LA_LEARN=0`), driven by `trace_drive_edge0.py`
(the 44 decode prompts of `trace_prompts.py`, 450 tokens each; 22 prompts, 3.5 GB, ~10K decode tokens).  `router_weights_edge0.py` extracts the 48 routers from
`model-skeleton.gguf` (512 x 2560, BF16).  Check: W_l x_l reproduces the recorded routing (rank 0 99.87%, top-10 overlap 99.69%).
Scripts use the system `python3` (numpy, torch CPU, gguf).

Results, held-out windows (the last 30%), true rank 0-1 experts inside the predicted top-4 (`pred_unify_edge0.py`):
| | lead 1 | lead 2 |
|---|---|---|
| router reuse (the engine's look-ahead before learning) | 74.9 | 58.2 |
| learned full-size correction on the hidden state (ridge, starts at the router) | 86.2 | 74.5 |
| + this token's previous-layer picks | 86.7 | 75.1 |
| + the previous token's routing (an oracle: causal only for a window's first row) | 86.8 | 75.4 |
| rank 16 / rank 64 of the same correction | 80.6 / 82.7 | 66.0 / 68.7 |
Token-ahead inputs add <= 0.3 points; the hidden-state correction is the gain, and it needs high rank.

`sim_follow_edge0.py <trace> <resident per layer> <lead>`: with 25 resident experts per layer (frequency-chosen), the share of the ten picks on the card (hit) against the
router weight on picks that were changed (swapped).  The stand-in rule (router weight x logit gap <= budget) and a router bias toward available experts reach the same
hit for the same swapped weight (bias is not better).  Making the pre-router's top-K ready for the token: exact hit (no swaps) at lead 1, K=4: 42.0 (router reuse) ->
43.5 (pre-router); K=8: 65.5 -> 69.9; lead 2, K=4: 35.8 -> 39.0.
