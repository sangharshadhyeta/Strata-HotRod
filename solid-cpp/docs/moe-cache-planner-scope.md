# Scope: MoE-cache pre-flight planner

Status: **implemented** (`--fit-params-moe-cache`, `common_fit_print_moe_cache()`
in `common/fit.cpp`) - this doc's "scoping only" status line went stale after
the fact; items 1-4 below (expert-size scanner with the widened fused-tensor
name check, GPU-count mode decision, budget/reserve sizing, MAX_BATCH
recommendation) are real, working code, confirmed by reading it directly
2026-08-14, not a stub. Item 5 (naive hit-rate floor) and the CPU-offload-
vs-partial-placement preference heuristic (adapted from leloch's removed
`common_moe_cache_prefers_cpu_moe()`) are implemented too. Written
2026-08-13, follows directly from the RTX 3060 validation (see
moe-cache-colibri-notes.md) where two undocumented defaults silently blocked
the cache and had to be found by adding temporary diagnostics to the source.
This tool exists to compute the correct values ahead of time instead of
repeating that debugging marathon on the H200 deployment.

## Where it lives - extend, don't create a new binary

`tools/fit-params/fit-params.cpp` (binary `llama-fit-params`) already does
almost everything this needs:

- Full CLI arg parsing via `common_params_parse(..., LLAMA_EXAMPLE_FIT_PARAMS)`
  - already understands `-ncmoe`/`--cpu-moe`/`--moe-cache`/`-ngl`/`-c` etc.
    since it shares the same `common_params` struct as every other tool.
- `common_get_device_memory_data_impl()` (`common/fit.cpp`) - loads the
  model with `no_alloc=true` and returns, per device: free/total VRAM bytes,
  and a memory breakdown (model/context/compute bytes) for the CLI args as
  given. Also returns `hp_n_expert` (expert count from hparams) as an
  out-param. This is a real no-inference, metadata-only load - matches
  exactly the "static, no running server" planning model we want (see the
  static-vs-dynamic distinction in the main notes doc).
- Two existing modes already: print fitted CLI args (`-c N -ngl N -ts ... -ot
  ...`), or print raw memory-in-MiB (`--fit-params-print`).

New flag: `--fit-params-moe-cache` (name tentative). Third mode, following
the existing pattern rather than overloading the other two. Keeps the diff
small and opt-in, matching the maintainer preference we saw directly in the
RFC thread (#21961: "paged attn... we can obviously implement the same
scheduling... at server level" - narrower, incremental, reuses what exists,
not a rewrite).

## What doesn't exist yet and needs new code

### 1. Per-expert-tensor byte size scanner

This is the one genuinely new piece. It briefly existed in our own git
history - `common_moe_cache_expert_kib_in_file()`, added in commit
`e9de20cf1` (ported from leloch's intermediate design), removed by the final
rework `1955dce6f`. The old version:

```cpp
static long common_moe_cache_expert_kib_in_file(const char * path, int64_t n_expert) {
    struct gguf_init_params ip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(path, ip);
    if (!gctx) return -1;
    long kib = -1;
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gctx, i);
        if (!strstr(name, "ffn_up_exps") && !strstr(name, "ffn_gate_exps")) continue;
        if (n_expert > 0) {
            kib = (long)(gguf_get_tensor_size(gctx, i) / n_expert / 1024);
        }
        break;
    }
    gguf_free(gctx);
    return kib;
}
```

**Cannot reuse as-is.** It only checks `ffn_up_exps`/`ffn_gate_exps`
(separate-tensor architectures). Both our actual target models use the
*fused* `ffn_gate_up_exps` tensor instead (confirmed directly: Gemma-4-A4B
via `gemma4.cpp`, GLM-5.2 via `glm-dsa.cpp` - also DeepSeek2, Cohere2MoE,
Qwen3.5MoE per the earlier grep). Needs a widened name check:
`ffn_gate_up_exps` OR `ffn_up_exps`/`ffn_gate_exps` OR fall back to
`ffn_down_exps` if neither gate variant is found. Given llama.cpp supports
15+ MoE architectures with potentially other naming conventions not yet
checked, this should degrade gracefully (return -1/unknown rather than
crash or silently mis-measure) on an unrecognized layout, and the planner's
output should say so plainly rather than emit a confident-looking wrong
number.

### 2. GPU-count-based mode decision

If `devs.size() == 1`: always recommend an explicit numeric
`--moe-cache <MiB>`, never `auto`/`on`. Print the reason, not just the
recommendation - cite the actual code condition
(`automatic && budget_devices < 2` in `moe-cache.cu`) so the warning is
verifiable, not just asserted.

### 3. Budget / `RESERVE_MB` sizing

Reuse the memory breakdown `common_get_device_memory_data_impl()` already
computes for the user's actual requested config (`dmd[id].free`,
`dmd[id].mb.model/context/compute`) to get expected-free-after-load. Then:

- `RESERVE_MB`: a modest safety margin, not the 3072 default. Exact sizing
  policy TBD during implementation (candidates: fixed small value like 256,
  a percentage of expected-free, or scaled by how far expected-free is
  above the minimum pool requirement) - needs a few real test runs across
  different VRAM sizes to pick a policy that doesn't need per-card tuning.
- `--moe-cache` budget: most of the remainder after reserve, **clamped to
  clear the `64 * expert_size` minimum pool requirement with an explicit
  warning if it can't** - don't silently emit a config doomed to the same
  "eligible_devices=0, nothing happens, no error" failure we hit on the
  RTX 3060. This is the single most important thing this tool exists to
  prevent.

### 4. `MAX_BATCH`

No computation needed - always recommend/emit
`GGML_CUDA_MOE_CACHE_MAX_BATCH=8`. The default of 1 is simply wrong for any
real workload (rejects everything except exact single-token batches,
including ordinary prefill).

### 5. Naive hit-rate floor projection (optional, Colibri-inspired, lower priority than 1-4)

`resident_bytes / total_expert_bytes`, matching Colibri's
`resource_plan.py` approach. Must be printed with an explicit caveat, not
as a confident promise: this is a capacity-ratio floor. Real hit rate can
be much higher on skewed-routing models (confirmed for both our targets:
GLM-5.2 and Gemma-4-A4B) or roughly match the floor on uniform-routing
models (DeepSeek's aux-loss-free routing, per the PR #26563 thread
finding). Cite the routing-skew section in the main notes doc rather than
re-explain it inline.

## Output format (v1: human-readable text, matching existing tool style)

Sketch, not final:

```
--moe-cache 8192
GGML_CUDA_MOE_CACHE_MAX_BATCH=8
GGML_CUDA_MOE_CACHE_RESERVE_MB=512
# projected hit-rate floor: 34.2% (capacity-ratio estimate; likely higher on
#   skewed-routing models like this one - see docs/moe-cache-colibri-notes.md)
# WARNING: 1 GPU detected - auto/on modes never engage the cache on a single
#   GPU regardless of free VRAM. Explicit budget (above) is required.
```

JSON/machine-readable output deferred to v2 if there's demand - v1 stays
human-readable only, matching the existing tool's convention, to keep the
diff small and reviewable.

## Explicitly out of scope for v1

- RAM/disk-tier planning (Colibri's 3-tier design) - not needed; our own
  capacity math already shows the GLM-5.2/H200 target fits entirely in RAM
  with no disk streaming required.
- Auto-tuning *other* features based on projection (Colibri's "genuinely
  novel bit" - e.g. disabling speculative decoding when projected hit rate
  is low) - real idea, bigger scope, v2 at earliest.
- NUMA-awareness, SMT-correct physical core counting - v2, not blocking for
  a single-socket H200 node.
- Actually launching a run with the computed flags - v1 only prints/
  recommends, matching the existing tool's "print fitted args, user copies
  them" convention. No auto-invocation.
- Multi-GPU pool-distribution planning (sticky layer/device routing across
  cards) - not relevant to the single-H200 target; would need its own scope
  if a multi-GPU deployment ever comes up.

## Relationship to the existing `--fit-target` mechanism

Complementary, not competing. `--fit-target`/`common_fit_params` decides
*placement* - how many layers fit in VRAM at all, `-ncmoe` sizing, tensor
split across multiple GPUs. This tool assumes placement is already decided
(user ran `--fit-target` first, or set `-ncmoe`/`--cpu-moe` manually) and
computes the *moe-cache tuning* for whatever VRAM is left over after that
placement. Natural workflow: resolve placement first (existing mechanism),
then run this tool's moe-cache recommendation against the same resolved
`mparams`/`cparams`, reusing the same `common_get_device_memory_data_impl`
call convention for consistency between the two.

## Testing plan (before considering this done)

1. Unit tests for the expert-size scanner across different GGUF tensor
   naming conventions (fused `ffn_gate_up_exps` vs separate
   `ffn_gate_exps`/`ffn_up_exps`) - at minimum, synthetic GGUFs matching
   both patterns; `tests/test-llama-archs.cpp`'s tiny-synthetic-model
   generator (found earlier this session, per-architecture, no download
   needed) could plausibly produce fixtures for this without needing real
   multi-GB model files.
2. End-to-end validation against the RTX 3060 + Gemma-4-26B-A4B case we
   already have real numbers for: does the planner's recommended
   `--moe-cache <N>` / `RESERVE_MB` reproduce (or beat) the hand-tuned
   41.6 tok/s / 62.1% hit-rate result from the validated run?
3. Cross-check the scanner's output against the GLM-5.2 GGUF header values
   we already hand-verified via `gguf_dump.py` earlier this session
   (`glm-dsa.expert_count = 256`, `glm-dsa.block_count = 79`) as a sanity
   check before trusting it on the real H200 deployment.

## Open question not yet resolved

Exact `RESERVE_MB` sizing policy (fixed vs. percentage vs. scaled-by-margin-
above-minimum) needs empirical tuning across at least two different VRAM
sizes (the RTX 3060's 12GB and the H200's 141GB) before picking one -
don't guess a formula and ship it unvalidated, given how much damage a
wrong default already did on this exact knob.
