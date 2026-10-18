# Expert predictor experiment (18 Oct 2026)

Question: can a predictor tell, a layer or two ahead, which experts a token's rank 0-1 picks will be, so they are on the GPU before the CPU has to compute them?
Model and hardware: Qwen3.8-Flash-Next IQ3_S on the RTX 3060 (12 GB).  The CPU rank 0-1 path is NOT removed; the predictor only makes it needed less often.

## How the data was made
* `STRATA_ROUTER_TRACE=<file>` (src/core/expert_source.cpp, `expert_pool_dispatch_multi`; off by default) writes, for every MoE layer of every decode window, the router input (bf16)
  and the routed ids.  Run with the stand-ins off so the ids are the model's own top-10.  `trace_run.py` starts a trace engine on port 8097 and drives it with
  `trace_prompts.py` (44 synthetic prompts: code, agent, reasoning, prose, other languages, summaries).  Result: 24,708 decode tokens x 48 layers (6.1 GB).
* Check: `W_l @ x_l` reproduces the recorded routing (rank 0 agrees 99.3%, top-10 overlap 99.5%).

## Results (true rank 0-1 experts found inside the predicted top-K; `*.out` are the raw outputs)
| | lead 1 layer, K=4 | lead 2, K=4 | lead 3, K=4 |
|---|---|---|---|
| router-reuse (layer l's real router on x_{l-d}; what the RouterLookahead does) | 74.4 | 64.3 | 59.3 |
| ridge trained on our trace, same prompts held out (inflated by same-prompt neighbours) | 89.2 | 85.2 | 82.5 |
| ridge trained on our trace, shifted text (other languages, summaries) | 74.7 | 67.5 | 63.5 |
| **published per-layer linear model (aswinkumar99/qwen3.8-flash-next-expert-traces, NVFP4), zero-shot** | **89.4** / shifted 85.5 | n/a (d=1 only) | n/a |
| the same, fine-tuned on our trace | 93.5 / shifted 85.9 | | |

* The published model transfers from NVFP4 to IQ3_S with no loss and generalises across text types; our 25k-token ridge does not.
* Live learning (`pred_online.py`, test-then-train, mini-batches of 16): started at the real router it is safe and gains 8-10 points at lead 2 within ~1,000 tokens; at lead 1 it adds ~0
  in 24k tokens.  Started from ZERO (also solid.cpp's rule) it stays below the baseline for the whole stream.
* `pred_cpu_proxy.py`: with a frequency-chosen hot set of 58 experts per layer, admitting K predicted experts per layer avoids this share of the CPU rank 0-1 computations:
  K=2: 71.5% (published model) vs 62.2% (router-reuse) vs 1.0% (just cache K more frequent experts); K=4: 87.1 / 79.9 / 1.9.  An upper bound: it assumes the expert arrives in time and finds a slot.

## Next
Fold the published d=1 models (standardisation into weight and bias) into the GPU look-ahead node as the matrix for the next layer, measure live recall and lateness in shadow mode,
only then admit (rank 0-1, a few slots, confidence-gated, free slots first), with the CPU path unchanged.  Lead 2+: start from the real router weights and learn online.

The published `.pt` files are pickles: load with `torch.load(weights_only=True)` only (their globals were checked: OrderedDict, FloatStorage, _rebuild_tensor_v2).
Dataset licence CC-BY-4.0 (attribution: aswinkumar99, qwen3.8-flash-next-expert-traces).
