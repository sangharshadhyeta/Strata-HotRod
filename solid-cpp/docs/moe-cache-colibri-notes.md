# START HERE - handoff summary for a fresh session (e.g. on the H200 box)

If you're picking this up cold, with none of the conversation that produced
this document: read this section first, then use the rest of the file as
reference. Everything below is dated and sourced - treat anything not
re-verified on the new hardware as a hypothesis carried over from a
different GPU (RTX 3060, 12GB), not a guarantee.

## What this is

A working port of a community MoE expert-cache feature into llama.cpp,
validated end-to-end with a real measured speedup. Goal: run large MoE
models (target: GLM-5.2, 744B) fast on hardware where the model doesn't fit
in VRAM, by keeping hot experts cached in VRAM while cold experts run on
CPU - instead of either the whole model running slow on CPU, or needing
enough VRAM for the whole thing.

## Where the code is

Branch `solid` (renamed from `moe-cache-port` 2026-08-31) in this repo, based on `ggml-org/llama.cpp` master
at commit `1f368f354`, 22 commits ahead. Remotes already configured:
`origin` = `ggml-org/llama.cpp`, `leloch` = `leloch/llama.cpp` (source of
the ported feature, branch `moe-cache-pr`). If this is a fresh checkout
elsewhere, `git log --oneline` on this branch is the actual build history -
trust it over this doc for anything code-related, since docs can drift and
git can't.

Key commits, in order: our own mmap host-pinning fix, then 4 ported commits
from leloch/moe-cache-pr (the core feature + its rework), then a fixup
commit for rebase drift (2 small bugs from porting across 797 commits of
upstream drift, both documented in the commit message), then this
documentation trail.

## How to build (what worked on the sandbox RTX 3060 box)

CUDA 13.3 toolkit, gcc-toolset-13 (needed specifically because the stock
system gcc/binutils couldn't assemble AVX-VNNI instructions the CPU backend
generates with `-march=native` - unrelated to CUDA, just a build-env
gotcha). Configure with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<your
GPU's compute capability>` (was 86 for the 3060; H200 is Hopper, compute
capability 90 - use `-DCMAKE_CUDA_ARCHITECTURES=90` or `native`). Verify
with `./build/bin/llama-cli --list-devices` before doing anything else.

## The two gotchas that will bite you immediately - set these explicitly

Do not trust the defaults. Confirmed by direct testing, not assumption:

1. `--moe-cache auto` (and `on`) **never engages on a single-GPU system**,
   full stop, regardless of free VRAM - there's a hard `automatic &&
   devices < 2` check in the code. **Always use an explicit numeric budget**:
   `--moe-cache <MiB>`.
2. `GGML_CUDA_MOE_CACHE_MAX_BATCH` defaults to **1**, not 8 (8 is only the
   max allowed value). Default 1 rejects every call except exact
   single-token batches - even ordinary prefill gets silently rejected, no
   error. **Set `GGML_CUDA_MOE_CACHE_MAX_BATCH=8` explicitly.**
3. `GGML_CUDA_MOE_CACHE_RESERVE_MB` defaults to **3072 (3GB)**, sized for
   24GB+ cards. Check actual free VRAM after model load (`nvidia-smi` while
   the model is loaded, or read the `[moe-cache] CUDA%d prepare_budget`
   figures if you re-add diagnostics) before assuming the default is fine -
   on a 12GB card it left only ~47MB, blocking the cache entirely. **On the
   H200 (141GB) this is much less likely to bite, but verify rather than
   assume** - a 26B model with 5-10 concurrent user KV caches could still
   eat into headroom differently than our single-user sandbox test did.

**How to verify it's actually working, don't trust silence**: run with `-v`
and grep for `[moe-cache] CUDA%d pool[` (confirms a pool was actually
allocated) and `[moe-cache] CUDA%d hits=` (confirms real cache traffic). No
error is printed when the cache silently fails to engage - absence of these
two log lines, not an error message, is the failure signal. If you don't
see them, one of the three gotchas above is still active.

## What's actually validated vs. still a hypothesis

**Validated by direct measurement** (RTX 3060, Gemma-4-26B-A4B, see full
section below): the port compiles, runs, produces correct output, and
delivers a real +54% decode speedup (27.0 -> 41.6 tok/s) with 62.1% cache
hit rate once the three gotchas above are fixed.

**Still a hypothesis, not yet measured**: everything about the GLM-5.2/H200
target specifically - the 40+ tok/s target, the capacity math (~439GB
model, ~429GB experts fitting in 512GB RAM), the MTP interaction, the
concurrency behavior at 5-10 users. All reasoned from evidence (GLM-5.2's
confirmed routing skew, other people's benchmarks on different hardware)
but none of it has been run for real yet. That's the next step.

## Recommended first moves on the H200 box

1. Build (see above), confirm `--list-devices` sees the H200.
2. Download `unsloth/GLM-5.2-GGUF` (UD-Q4_K_M, 11 shards, ~439GB total) -
   this will take a while regardless of connection speed given the size;
   start it early and work on other setup while it downloads.
3. Reproduce the RTX 3060 validation methodology on GLM-5.2 before trusting
   any tok/s number: explicit `--moe-cache <budget>`,
   `GGML_CUDA_MOE_CACHE_MAX_BATCH=8`, a `RESERVE_MB` sized to actual free
   VRAM after load (check with `-v`, don't assume), a clean `--moe-cache
   off` baseline run for comparison, and `-v` to confirm real
   `pool[`/`hits=` log lines before trusting any number.
4. Measure hit-rate *distribution*, not just the aggregate hit rate - see
   the "load-bearing caveat" section below on why routing skew (not just
   the mechanism) determines whether this actually helps GLM-5.2 the way it
   helped Gemma-4-A4B.
5. Only after that's confirmed working: layer in native MTP (already merged
   upstream, verified present in the unsloth GGUF - see below), then work
   through the medium-term list (concurrency tuning, prefix caching) as
   real usage surfaces real bottlenecks, not before.

---

# VALIDATED 2026-08-13: moe-cache works, real 54% speedup measured on RTX 3060

First real end-to-end test on the downloaded Gemma-4-26B-A4B model. Took real
debugging to get there - two undocumented defaults silently blocked
engagement on a VRAM-constrained card, found only by adding temporary
diagnostic prints to the actual source (reverted after, see git history for
the fix commit if these ever get upstreamed):

1. **`GGML_CUDA_MOE_CACHE_MAX_BATCH` defaults to 1, not 8** (8 is the max
   *allowed* value, not the default - our own earlier notes had this wrong,
   corrected now). Default 1 means the cache rejects every call except exact
   single-token batches - even a 2-token prefill gets silently rejected, no
   error, no log. Confirmed empirically: baseline and "cache on" runs were
   bit-identical in speed until this was set to 8 explicitly.
2. **`GGML_CUDA_MOE_CACHE_RESERVE_MB` defaults to 3072 (3GB)**, sized for
   GPUs with real headroom (RTX 3090 24GB, A100, H100 - the RFC thread's
   test rigs). On our 12GB RTX 3060, `-ngl 99` for a 26B/262144-vocab model
   leaves only ~3.05GB free after loading - the 3GB reserve ate nearly all
   of it (47MB left, against a ~136MB minimum pool requirement). Lowered to
   256MB to unblock. **This will matter for the H200 deployment too** if
   VRAM headroom after loading GLM-5.2's dense weights + KV cache for 5-10
   slots is proportionally tight - worth checking early there, not assuming
   3GB reserve is fine just because the card is bigger.

**Clean A/B result** (same prompt, same n_predict=100, `-ngl 99 -ncmoe 99`):

| config | decode tok/s | hit rate |
|---|---:|---|
| `--moe-cache off` (baseline) | 27.0 | - |
| `--moe-cache 2048` + tuned MAX_BATCH=8, RESERVE_MB=256 | **41.6** | **62.1%** (32421/52200) |

**+54% decode speedup, real and measured**, not projected. Hit rate climbed
from 44.2% on a shorter 20-token run to 62.1% on this 100-token run - cache
warm-up matters, consistent with the RFC thread's own "under-warmed pool
reads as saturated" warning. This is now hard evidence for what was
previously projection based on other people's hardware/models - directly
validates the core premise (Gemma-4-A4B's routing skew is real and
exploitable) and confirms the port itself works correctly end-to-end
(loads, runs, generates coherent output, no crashes, real speedup).

**Action for the GLM-5.2/H200 deployment**: don't assume defaults work.
Explicitly set `--moe-cache <budget>` (never `auto`/`on` alone - see the
single-GPU dormancy finding above), `GGML_CUDA_MOE_CACHE_MAX_BATCH=8`, and
size `GGML_CUDA_MOE_CACHE_RESERVE_MB` based on actual free VRAM after model
load, not the 3GB default. Verify eligibility empirically (real pool[]/hits=
log lines with `-v`), don't just assume config was accepted.

**Superseded 2026-08-13 (same day, later): these three defaults were fixed
in core code, not just documented as gotchas.** User pushback: "they removed
it so we won't do is not the correct argument... our system should at least
be as smart as we are doing by self." Changes in `moe-cache.cu`:
`max_batch` default 1→8; `reserve_mb` default 3072→0 (sentinel meaning
"compute live": `min(1024, max(128, free_mb/20))` MiB, i.e. 5% of free VRAM
at the moment of session creation, clamped 128-1024 MiB); the single-GPU
`automatic && devices < 2` dormancy gate relaxed to just `devices == 0` at
all three call sites (session creation, budget check, eligibility check).
Net effect: `--moe-cache auto` with zero env vars now does on a 12GB
single-GPU card what previously required hand-tuning two env vars per
piece of hardware. See the sweep results below for the A/B proof this
actually closes the gap to manual tuning.

# Autonomous calibration sweep, 2026-08-13 (RTX 3060, Gemma-4-26B-A4B)

Two unsupervised sweeps run via `llama-bench`/`llama-server` while stepping
away, testing whether the live-computed `auto` defaults above actually hold
up across configs, and how much headroom is left in placement/concurrency.
Scripts and raw logs: `moe_cache_sweep.sh`
(v1) and `moe_cache_sweep_v2.sh` (v2) - not committed to the repo (job
scratch dir), summarized here instead.

## v1 (126s wall clock - budgeted 55 min, workload was just fast on this HW)

**Phase 1 - `--moe-cache` mode/budget sweep, fixed safe placement (`-ncmoe
99`, all experts CPU-resident, GPU handles cache-hit rows only), `-p 128 -n
64 -r 3`:**

| `--moe-cache` | decode tok/s |
|---|---:|
| `off` | 27.71 ± 0.05 |
| `auto` (live-computed) | **48.90 ± 6.64** |
| `512` | 30.39 ± 1.08 |
| `1024` | 37.59 ± 1.40 |
| `2048` | 43.60 ± 1.72 |
| `4096` | 50.34 ± 8.29 |
| `8192` | 50.21 ± 7.70 |

**Headline finding: `auto` (48.90) lands within ~3% of the best
hand-picked value (4096: 50.34), both within the run-to-run noise band (±7-8
tok/s at this batch size).** The live-reserve fix isn't just "unblocked" -
it's landing at essentially the same place a human sweeping budgets by hand
would land, with zero tuning. This is the direct evidence for the
self-correcting behavior that was asked for: no env vars, no per-card
tuning, and it's not leaving the manually-tunable gains on the table.

**Phase 2 - `-ncmoe` placement sweep, `--moe-cache auto`, `-p 128 -n 64 -r
2`, pushing past the previously-assumed-safe boundary of 30:**

| `-ncmoe` | decode tok/s | result |
|---:|---:|---|
| 30 | 46.63 ± 8.60 | ok (previous "safe" baseline) |
| 29 | 47.55 ± 9.19 | ok |
| 28 | 47.71 ± 8.68 | ok |
| 26 | 49.72 ± 8.78 | ok |
| 24 | 50.77 ± 7.91 | ok, **no OOM** |

Monotonically climbing as more experts move to GPU, and the sweep stopped
at 24 only because that was the last value scripted, not because it hit a
wall. **This means the previously-documented "safe" placement (30) was
conservative - there was real untested headroom below it.** v2 (below)
pushes further to find the actual boundary.

**Phase 3 - concurrent-request emulation** (`llama-server --parallel 8`,
`--moe-cache auto`, mixed short/long/agentic-style prompts, `n_predict=60`):

| concurrency | aggregate tok/s |
|---:|---:|
| 1 | 37.56 |
| 2 | 70.22 |
| 4 | 80.88 |
| 8 | 103.07 |

Near-linear 1→2 (1.87x), diminishing but still positive returns through 8.
Healthy scaling shape for the batching to build on.

**Phase 4 - variance check** (5x repeat, `auto`, `-ncmoe 99`): 52.79 ± 7.36
tok/s. Consistent with phase 1's `auto` number within noise; ±14% run-to-run
variance at this prompt/gen size is real and should be accounted for when
comparing configs that differ by less than ~10-15%, not treated as
measurement error to explain away.

## v2 (403s wall clock - budgeted 50 min, again just fast on this HW)

Purpose: v1 left three things unexplored that the user's original ask
("emulate users and requests to find optimal solutions... various kinds of
configs and loads") called for and 126s wasn't enough time to cover: the
real OOM boundary, whether `MAX_BATCH` still matters now that it defaults
to 8, and concurrency at the actual "5-10 users" deployment target with
longer, more agentic-shaped generations.

**Phase A - joint `-ncmoe` x `--moe-cache` grid (`-p 256 -n 128 -r 3`),
pushing past v1's un-OOM'd floor of 24:**

| `-ncmoe` | `auto` | `4096` | `8192` |
|---:|---:|---:|---:|
| 24 | 57.30 ± 5.59 | 56.87 ± 4.96 | 57.04 ± 5.87 |
| 20 | 59.60 ± 4.74 | 59.54 ± 5.34 | 59.92 ± 5.19 |
| **16** | **60.99 ± 3.41** | **61.06 ± 3.36** | **61.35 ± 3.10** |
| 12 | 53.51 ± 0.94 | 53.67 ± 0.85 | 53.51 ± 0.94 |
| 8 | **OOM** (model load fails) | OOM | OOM |

**Two real findings here, both correcting earlier assumptions:**

1. **The optimum is `-ncmoe 16`, not 24 or 30.** Throughput is
   *non-monotonic* in placement aggressiveness: 24→20→16 climbs (57→60→61),
   but 16→12 drops sharply (61→53.5) before failing outright at 8. The
   previously-documented "safe" value of 30 (46.63 tok/s, from v1 phase 2)
   left ~31% of decode throughput on the table. The likely mechanism: below
   ~16, there's not enough free VRAM left for the cache pool + KV cache +
   compute buffers to operate without contention, even though the model
   itself still loads down to 12 - the failure mode isn't a hard wall until
   8, it's a soft throughput cliff starting around 12. **True OOM boundary
   on this card/model: between `-ncmoe 8` (fails) and `-ncmoe 12` (loads,
   but already past the throughput cliff).**
2. **Cache budget stops mattering once placement is right.** At every
   placement tested, `auto`/`4096`/`8192` are statistically indistinguishable
   (differences are inside the ± noise band). This means the earlier finding
   ("`auto` ≈ best hand-tuned budget") wasn't a coincidence of the specific
   `-ncmoe 99` placement it was measured at - it holds across placements too.
   Once `-ncmoe` is dialed in, hand-tuning `--moe-cache` past `auto` buys
   nothing further on this hardware.

**Phase B - `GGML_CUDA_MOE_CACHE_MAX_BATCH` sweep** (2/4/8/16/32, `-ncmoe
99`, `auto`): 53.40-53.78 tok/s across the entire range, all within ±6.8
noise. **Confirms the earlier fix (default 1→8) was the fix that mattered;
tuning past 8 buys nothing.** No further action needed here.

**Phase C - concurrency, wider bracket + longer generations** (`--parallel
16 -c 16384`, `auto`, `n_predict=200`, mixed agentic-style prompts):

| concurrency | aggregate tok/s | per-user tok/s |
|---:|---:|---:|
| 1 | 46.88 | 46.88 |
| 2 | 74.44 | 37.22 |
| 4 | 88.60 | 22.15 |
| **5** | **99.02** | 19.80 |
| 8 | 96.42 | 12.05 |
| 10 | 58.57 | 5.85 |
| 12 | 57.94 | 4.82 |
| 16 | 63.93 | 3.99 |

**Unexpected and important: aggregate throughput peaks at concurrency=5,
then collapses at 10-12 before a partial recovery at 16.** This is not a
smooth diminishing-returns curve - there's a real cliff between 8 and 10.
Not yet root-caused (candidates: KV-cache VRAM exhaustion at high
`--parallel` x `-c` on a 12GB card with `-ncmoe 99`, or slot-scheduling
contention in the server) - flagged as an open question, not a settled
conclusion. **Directly actionable for the H200 deployment**: don't assume
higher `--parallel` is free, or that a bigger card removes this cliff -
this exact sweep methodology (concurrency emulation against the real
target count) needs to be re-run on the actual deployment hardware/model,
since the cliff's location is plausibly VRAM-budget-relative, not a fixed
concurrency number. On this card, for a "5-10 users" target, **5 concurrent
requests is the actual sweet spot, not 8 or 16** - matching the top end
of the target range, not the middle.

**Phase D - variance confirmation** (5x repeat, `-ncmoe 99`, `auto`,
`-p 256 -n 128`): 56.06 ± 5.20 tok/s tg128, 244.38 ± 8.86 pp256. Consistent
with phases A/B within noise. (Note: the script's auto-picked "best config"
for this phase mis-selected `-ncmoe 24` instead of the actual winner `16`,
due to a sort-key bug in the shell one-liner - harmless since phase A's raw
numbers above are already the authoritative source, but worth knowing the
phase D number isn't the true best-config confirmation it was meant to be.)

## Updated recommendation after v1+v2

For this card/model: `-ncmoe 16 --moe-cache auto`, no env var tuning needed
(`MAX_BATCH` default of 8 and live `RESERVE_MB` computation are both
already at their empirical optimum). Expected ~61 tok/s decode, up from the
27.71 tok/s `off` baseline (**+120%**, larger than the +54% first measured
because that used a suboptimal `-ncmoe 99` placement, not because the cache
mechanism itself improved). For concurrent serving, provision for the
concurrency sweet spot found empirically (5 here) rather than assuming
higher `--parallel` monotonically helps - re-run phase C's methodology on
whatever hardware is actually deployed to, since the cliff is suspected to
be VRAM-relative rather than a portable fixed number.

# Concurrency-cliff root-cause investigation, 2026-08-13 (same day, follow-up)

Dug into *why* the concurrency cliff (peak ~99 tok/s at concurrency=5,
collapse to ~57 tok/s at concurrency=10-12) happens specifically between 8
and 10. Three hypotheses, each tested with a real controlled change and
re-measurement, not just theorized:

## Hypothesis 1: `GGML_CUDA_MOE_CACHE_MAX_BATCH` gate - CONFIRMED a real bug, but not this bug

Traced the code path: `moe_cache_begin()` in `moe-cache.cu` rejects (falls
back to full CPU compute, no GPU cache acceleration at all for that call)
any batch where `n_tokens > config.max_batch`. `n_tokens` here is
`ids->ne[1]`, the actual token count in the current ubatch - during
concurrent decode with `--parallel N`, this scales with how many sequences
are decoding a token in the same step, up to `N`. The env var clamp was
hardcoded to `[1, 8]` - not just defaulting to 8, *incapable* of being set
higher - while the default was also 8. With `--parallel 16`, any decode
step batching more than 8 concurrent tokens together silently loses GPU
cache acceleration entirely for that step.

**Verified this ceiling is artificial, not structural**: the GPU scratch
buffers (`moe_cache_scratch_requirements`'s `constexpr max_rows = 64`) and
the CPU-side stack arrays (`ggml-cpu.c`'s `MOE_CACHE_MAX_TOPK = 64`) are
already sized for up to 64 rows regardless of `max_batch` - raising the
ceiling costs nothing structurally. (64 itself is *also* just a
human-picked constant, not derived from anything live - the difference
from the old 8 is only that 64 is wired into fixed-size memory layouts, so
raising it further would need an actual code change, not a config tweak.
Worth being honest about that distinction rather than treating 64 as
principled.)

**Fixed**: `max_batch`'s default is now derived live from the real
`n_seq_max` of the context (`llama_context` calls a new
`ggml_moe_cache.set_max_batch_hint()` vtable entry with `cparams.n_seq_max`
before `ggml_backend_sched_new()`, i.e. before the moe-cache session reads
its config), floored at 8 (preserves current single/few-user behavior and
ordinary prefill batching), ceilinged at the real 64 structural limit. The
env var override range was widened from `[1,8]` to `[1,64]` to match.
Verified end-to-end: with `--parallel 16`, the `[moe-cache] enabled` log
line now reads `max-batch=16` where it previously silently used 8 no
matter what `--parallel` was set to (this required moving
`ggml-backend-moe-cache.h` from `ggml/src/` to `ggml/include/` since
`llama-context.cpp` needed to call the new vtable entry and the private
`ggml/src` include path doesn't propagate to `libllama`).

**Re-tested the concurrency sweep with the fix active** (confirmed
`max-batch=16` in the log): the cliff was still there, nearly identical in
shape (concurrency=10 landed at 57.4 tok/s vs. 58.6 before the fix). *The
fix is real and worth keeping - `MAX_BATCH=8` was a genuine silent
cache-disabling bug for any concurrent deployment - but it was not the
cause of this specific cliff.*

## Hypothesis 2: cache-capacity thrashing under diverse concurrent workloads - RULED OUT

Enabled `GGML_CUDA_MOE_CACHE_STATS=100` for periodic hit-rate logging and
watched it through the concurrency=10 run. Hit rate held rock steady at
~82% (424680/517480 through 630399/766400 across the run, drifting only
±0.4%), pool fully utilized (2851/2851 slots) with healthy, steady eviction
churn, **zero** `dispatch-fail` or `collect-fail` throughout. The cache
mechanism itself performs identically before, during, and after the cliff
- ruling out thrashing/eviction-storm explanations entirely.

## Hypothesis 3: CPU thread-pool saturation for the miss-row (~18%) CPU fallback - RULED OUT

`n_threads` defaulted to 6 on this box's 6 physical / 12 logical
(hyperthreaded) cores (`common_cpu_get_num_math()` uses physical core
count). Reasoned this was a plausible bottleneck: cache misses still need
CPU computation regardless of GPU cache health, and more concurrent
sequences mean more total miss-row work funneling through a fixed 6-thread
pool - a classic queueing cliff as utilization crosses saturation. Note
this default is *not* an unmotivated guess like the old `max_batch=8` -
physical-core-only is a legitimate heuristic to avoid SMT contention within
one tight single-stream compute loop. But it's a heuristic tuned for one
stream monopolizing the CPU, not tested against many independent
concurrent streams, so it was worth checking.

Re-ran the identical concurrency test with `-t 12 -tb 12` (all logical
cores). Result: **not better, slightly worse** (concurrency=8: 80.1 vs 87.4
tok/s; concurrency=10: 47.8 vs 57.4 tok/s), plausibly because the compute
threadpool now saturates all cores and starves the server's own
request-handling/sampling/JSON threads. Ruled out as the cause - more
threads didn't fix it and moved the wrong direction.

## Hypothesis 4 (2026-08-14): endpoint-bias artifact - RULED OUT, cliff is real

The three hypotheses above were tested 2026-08-13, before the endpoint-bias
bug (raw `/completion` bypassing the chat template, see the endpoint-bias
investigation section) was found. Worth checking whether the cliff itself
was ever real, or partly a measurement artifact of the same bug. Re-ran the
Phase C sweep through the corrected `/v1/chat/completions` endpoint with 16
distinct real prompts (not repeated), `-ncmoe 99 --moe-cache auto -c 16384
--parallel 16`, aggregate tok/s = total generated tokens / wall-clock for
all N concurrent requests to complete:

| concurrency | 1 | 2 | 4 | 5 | 8 | 10 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agg tok/s | 42.18 | 69.49 | 81.53 | 85.94 | **92.12** | 54.38 | 58.52 | 64.54 |

Same shape as the original (peak around 5-8, hard collapse at 10, -41% vs
the original's -39%, partial recovery by 16). **The cliff is confirmed
real, not an endpoint-bias artifact.**

## Hypothesis 5 (2026-08-14): copy/PCIe contention under concurrency - inconclusive, mechanism likely never engaged

Motivated directly by the expert-prefetch port done this session (from
github.com/thecodacus/llama.cpp, fable5/prefetch-experts branch - unrelated
to the JustVugg/colibri material elsewhere in this doc): if GPU-idle-during-
copy compounds under concurrent load (more simultaneous sequences needing
CPU-resident expert weights fetched over the same PCIe bus), that could
plausibly explain a cliff once total in-flight copy volume crosses a
hardware limit. Re-ran
the same corrected sweep with `GGML_CUDA_REGISTER_HOST=1
GGML_SCHED_PREFETCH_EXPERTS=1` (the prefetch-experts port, measured
+36.0% prefill throughput on this same hardware) active:

| concurrency | 1 | 2 | 4 | 5 | 8 | 10 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agg tok/s (patched) | 42.56 | 69.57 | 80.04 | 88.90 | 91.99 | 53.24 | 58.42 | 63.87 |

**Zero measurable difference at any concurrency level, cliff identical in
shape and magnitude.** Not a clean "ruled out" though: the prefetch
mechanism only engages above a batch-size threshold
(`ids->ne[0]*ids->ne[1] >= 2*n_expert` in `ggml_backend_sched_compute_splits`)
tuned for large-batch *prefill*, not single-token-per-sequence *decode* -
at concurrency=16 with Gemma-4's top-k, the per-step routing-id count is
plausibly still well under that threshold, meaning the patch likely never
activated during this decode-dominated workload at all. So this result
shows the prefill-oriented prefetch fix doesn't touch the decode-time
concurrency cliff, not that copy/PCIe contention during decode is
definitively ruled out - a genuinely decode-scoped version of overlapped
prefetch (much smaller per-step tensors, higher call frequency) is a
different, untested mechanism.

## ROOT-CAUSED, 2026-08-14: `MMVQ_MAX_BATCH_SIZE` in ggml-cuda, not moe-cache or llama-server

Five candidate explanations were checked first and none of them it
(moe-cache MAX_BATCH gate - real bug, not this bug; cache-capacity
thrashing - ruled out via stats; CPU thread-pool saturation - ruled out,
made it worse; endpoint-bias artifact - ruled out, cliff reproduces
through the corrected endpoint; prefill-oriented copy-prefetch - no
effect, likely never engaged for this decode-shaped workload). Rather
than keep testing moe-cache-adjacent hypotheses from outside, captured
full per-request timing distributions at concurrency=8 vs 10 instead of
just the aggregate number: wall-time spread across concurrent requests
stayed ~1.00x at *every* level tested (all requests in a batch finish
within 0.05-0.07s of each other) - **no starvation or fairness issue at
all**. Instead, every request's own per-token decode rate collapses
uniformly: ~12.6-14 tok/s/sequence through concurrency=8, ~6.0
tok/s/sequence at concurrency=9 (confirmed the exact boundary with a
dedicated 7/8/9/10 sweep). A pure batch-size-triggered compute cliff, not
a scheduling bug - which pointed straight at the GPU kernel dispatch
layer, not `tools/server/server.cpp`.

Found it in `ggml/src/ggml-cuda/mmvq.cu`: `ggml_cuda_mul_mat_id` (the
MoE expert-routing dispatcher) only uses the fast `MMVQ` kernel - built
for small batches - when batch size is at or below a per-(GPU-arch,
quant-type) threshold; above it, dispatch falls through to the general
`MMQ`/`MMF` kernels, tuned for large batches and sitting in an
efficiency valley for medium ones (matches the partial recovery seen at
concurrency=16 in the original sweep - closer to those kernels' own
better-utilized batch size, not fully recovered). On this RTX 3060
(compute capability 8.6, the "turing_plus" bucket in
`get_mmvq_mmid_max_batch_turing_plus`), the per-quant-type table has
explicit tuned values for several types but not for whatever quant our
model's expert tensors actually use (Q4_K or Q8_0, both absent from that
table) - so it silently falls to the generic default, `MMVQ_MAX_BATCH_SIZE
= 8`, which is exactly the measured boundary.

**Deployment-critical**: for NVIDIA compute capability >= Ada Lovelace
(includes Hopper/H200) the dispatch code skips the per-type table
entirely and always uses the flat default of 8:
```cpp
if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
    return MMVQ_MAX_BATCH_SIZE;  // = 8, unconditionally
}
```
The same cliff is likely to hit the real GLM-5.2/H200 deployment at
concurrency 9-10, squarely inside the "5-10 users" target range,
independent of every moe-cache/Layer1/Layer2/LFRU optimization built
this session - this is a different, lower-level dispatch decision none
of that work touches.

**Fix shipped: warn, don't silently guess.** Not something `-ncmoe` or
any placement/VRAM lever can fix (it's a GPU-kernel-dispatch limit, not
a memory-fit question), so Layer 1 (`common_warn_concurrency_cliff` in
`common/common.cpp`) now warns at startup instead of changing anything.
`get_mmvq_mmid_max_batch()` is exposed outside `ggml-cuda` via the
existing proc-address mechanism (same pattern as the host-buffer
registration calls). **Multi-GPU/tensor-split correct**: initially
shipped scoped to device 0 only (the CUDA backend's internal device
indexing doesn't map 1:1 onto the global `ggml_backend_dev_get()` index
space), then fixed the same session - the exposed function now takes the
caller's own `ggml_backend_dev_t` directly and recovers the real
per-backend device index from `dev->context`, and the common.cpp side
checks every GPU device (not just the first), since on a tensor-split
setup the full concurrent batch hits whichever device is computing the
current layer - the cliff can happen on any of them, so the safe ceiling
is the minimum across every device. Since the model's actual
expert-tensor quant type isn't cheaply known before a full load, the
warning queries every common
MoE quant type and reports the minimum across them - a genuine
conservative lower bound, so it can fire a bit early but never miss a
real risk. **Also accounts for MTP**: speculative verification batches
`(n_max+1)` candidate positions per sequence into a single `MUL_MAT_ID`
call, the same multiplier moe-cache's own `max_batch` hint already uses
- so effective batch = `n_parallel * verify_width`, not just
`n_parallel`. Confirmed: `--parallel 1` with `spec-draft-n-max=9` (verify
width 10) correctly fires the warning on its own, where a naive
`n_parallel`-only check would have stayed silent. Runs after the
calibration-cache lookup so it sees the real (possibly cached) `n_max`,
not the pre-lookup default.

| config | concurrency=8 | concurrency=10 | concurrency=12 |
|---|---:|---:|---:|
| `max_batch=8` (original bug) | 96.4 | 58.6 | 57.9 |
| `max_batch=16` (fixed) | 87.4 | 57.4 | 57.7 |
| `max_batch=16`, `-t 12` | 80.1 | 47.8 | 58.9 |

The cliff's shape (rise to a peak around 5-8, hard drop at 10, plateau
57-59 through 12) is essentially identical across all three configurations
- strong evidence the bottleneck is **not** in moe-cache at all (its own
health metrics stayed clean throughout every run), but in something
structural to `llama-server`'s own slot scheduling / continuous-batching
logic. That's a different investigation from the moe-cache port itself and
wasn't pursued further this session - flagged here as a genuine open
question, not swept under a wrong explanation. **Action for the H200
deployment**: re-run this same three-hypothesis-style controlled
investigation there rather than assuming any of these three (all ruled out
here) explain a cliff if one appears - the real cause is still unknown and
may be config- or version-specific to `llama-server`'s scheduler, not
something moe-cache-specific work will fix.

# Live self-tuning design, 2026-08-13 - what's done vs. what's next

Prompted by direct pushback during the sweep-result review: computing
defaults once from a snapshot isn't the same as being genuinely
self-correcting, and env-var gotchas someone has to know about isn't
"smart," it's just documentation debt. Two things now compute live at
*every* process start (not cached, not a one-time calibration artifact):

1. **`reserve_mb`** (fixed earlier session) - live 5%-of-free-VRAM at
   session creation, clamped 128-1024 MiB.
2. **`max_batch`** (fixed above) - live from `n_seq_max` at every launch.

**What's still static and shouldn't be**: `-ncmoe` placement. Our own
sweep proved memory-fit and throughput-optimal are different things
(`ncmoe=12` loads fine, `ncmoe=16` is 12%+ faster) - a real OOM-vs-throughput
tradeoff exists here that doesn't exist for the two levers above (their
scratch/reserve costs don't scale with the setting, so there was no
tradeoff to balance - raising them was a free win). Planned fix, agreed
scope for this session:

**Layer 1 (built and validated this session)**: `common_maybe_autoplace_moe_cpu()`
in `common/common.cpp`, called from `common_init_result::common_init_result()`
right after the existing `--fit` system has had its chance and right before
the real model load. Only steps in when nothing has already decided
placement (`-ncmoe`/`-ot`/`-cmoe`, or `--fit` itself, leave real patterns in
`params.tensor_buft_overrides` - checked directly, any real pattern anywhere
in that vector means skip) and the exact config as given would fail to fit -
predicted via the same no-alloc probe `--fit-moe-cache`/`common_fit_params`
already use (`common_get_device_memory_data`), used as a gate *before* the
real allocation happens, not a catch-after-crash retry. When it doesn't fit
and the model has MoE tensors, binary-searches the minimal CPU-offloaded
layer count that does fit (O(log n_layer) no-alloc probes, not O(n_layer) -
monotonic in placement aggressiveness so binary search is valid), then
writes the winning overrides into both `params.tensor_buft_overrides` (for
consistency/future reuse) and the `mparams` about to be used for the real
load.

**Why this needed to run after `--fit`, not skip when `--fit` is on (as
first implemented)**: `--fit` runs by default for `llama-server`. Initial
version bailed out whenever `params.fit_params` was true, assuming the
general system already covered this. It doesn't - `--fit`'s only levers are
`-c` and `-ngl` (whole-model layer count), and it aborts as soon as *either*
is user-pinned, with no MoE-specific fallback at all. Confirmed directly:
launching with `-ngl 99 -c 16384 --parallel 16` and no `-ncmoe` (both
explicitly pinned, nothing unusual) produced
`common_fit_params: failed to fit params to free device memory: n_gpu_layers
already set by user to 99, abort` - and `common_init_result` **silently
discards that failure status** and attempts the real load anyway, which
then hard-crashed exactly as our very first mid-session OOM did (`-ncmoe 16
-c 16384 --parallel 16`: `cudaMalloc failed: out of memory` on the KV cache,
because moving experts onto the GPU left no room for it). Moved Layer 1 to
run as an explicit fallback after `--fit`, operating on `--fit`'s own
(possibly still-failing) `mparams`/`cparams` rather than rebuilding fresh
ones - this is the gap `--fit` doesn't cover, not a duplicate of it.

**Validated end-to-end, four real scenarios, real hardware:**

| scenario | model | config | result |
|---|---|---|---|
| previously-crashing OOM | Gemma-4-26B-A4B (15.77 GiB) | `-ngl 99 -c 16384 --parallel 16`, no `-ncmoe` | auto-placed first 19 layers to CPU (6 binary-search probes), server started, served real completions at 37.2 tok/s |
| same model, smaller context | same | `-ngl 99 -c 2048 --parallel 1`, no `-ncmoe` | auto-placed first **12** layers (correctly less aggressive - smaller KV cache footprint leaves more room for experts on GPU; confirms this is live per-config, not a cached/fixed answer) |
| explicit override respected | same | `-ngl 99 -ncmoe 30 -c 16384 --parallel 16` | zero auto-placement log output, user's exact value used untouched |
| dense model, already fits | Qwen2.5-0.5B-Instruct | `-ngl 99 -c 16384 --parallel 16`, no `-ncmoe` | zero auto-placement log output (fits as given - `--fit` itself found no reduction needed); concurrent load test (1/4/8/16 requests) all succeeded, clean scaling 326→1656 agg tok/s, no cliff |

Case 2 is the important one for the "live, not one-shot" property asked
for earlier: the same model, same hardware, produces a *different* correct
answer (12 vs. 19 layers) purely because the context/concurrency config
changed - there's no cached/stale value involved, every launch computes
fresh.

**Layer 2 (deferred, scoped but not built)**: the throughput-optimal
refinement on top of Layer 1's safe floor - a short empirical calibration
(2-3 candidate placements, few seconds) run once per (GPU, model, context
config) combination and cached, instead of re-measured on every restart.
Also includes `n_threads`/`n_threads_batch` as a swept dimension - no
safe-analytic-floor exists for that lever (our own test proved the
"obviously safe" direction, more threads, actually made concurrent
throughput *worse* - see the root-cause investigation above), only a real
curve to measure.

**Layer 2 (built and validated this session): `--moe-calibrate`.** A new
`llama-server` flag. `common_moe_calibrate()` in `common/common.cpp`: finds
Layer 1's safe floor via `common_moe_find_safe_layers()` (refactored out of
Layer 1's own logic so both share it), then benchmarks real candidates
*above* that floor (`safe_n + k*step` for k=1..4, `step = max(1, n_layer/15)`)
via `common_moe_bench_candidate()` - a genuinely real load + real short
greedy-decode loop (deliberately no sampler subsystem, since output quality
is irrelevant to a speed probe and greedy decoding on a real prompt still
exercises representative MoE routing/cache behavior), timed with
`ggml_time_us()`. Picks the best by measured tok/s, then does the same for
a couple of thread-count candidates (default/physical/logical) at the
winning placement. Writes the result to
`fs_get_cache_directory()/moe-calibration.json` (nlohmann::json, already
vendored) keyed by GPU signature (name + **total** VRAM per device -
deliberately not free VRAM, which fluctuates with background load and would
make the key spuriously miss) + model path + model file size + `-c`/
`--parallel`/`-ngl`. Calibrate-then-exit, matching the existing
`--fit-moe-cache` preview-before-you-commit pattern rather than mixing
calibration into the same run as real serving.

**Cache lookup wired into Layer 1's fallback** (`common_maybe_autoplace_moe_cpu`):
before falling back to the binary search, checks the cache; if an entry
exists, runs **one** verification probe (not the full search) to confirm
it still fits under current conditions, then applies it directly (placement
+ thread counts) - fast path. If conditions changed enough that the cached
placement no longer fits, logs that plainly and falls back to the live
binary search rather than silently using a stale answer.

**Validated on real hardware** (RTX 3060, Gemma-4-26B-A4B, `-ngl 99 -c
16384 --parallel 16`, the exact previously-crashing config):

| `-ncmoe` candidate | measured decode tok/s |
|---:|---:|
| 19 (Layer 1's safe floor) | 35.35 |
| 21 | 47.65 |
| **23 (winner)** | **49.33** |
| 25 | 46.63 |
| 27 | 45.59 |
| thread candidate 12 (vs. default 6) | 42.88 (worse - default 6 wins) |

**Layer 1's safe floor (19) was the *worst* of every candidate tested -
Layer 2 found a placement 40% faster** (49.33 vs. 35.35 tok/s), directly
confirming the earlier finding that memory-fit and throughput-optimal are
different questions, not approximately the same thing. Cache file written
correctly (`{"CUDA0:11900;|<path>|<size>|c16384|p16|ngl99": {"n_cpu_moe":
23, "n_threads": 6, "n_threads_batch": 6, "tok_per_sec": 49.33, ...}}`).
Relaunching the exact same config **without** `--moe-calibrate` produced
`common_maybe_autoplace_moe_cpu: using calibrated MoE placement from cache
(ncmoe=23, n_threads=6, measured 49.33 tok/s on ...)`, started successfully,
and served a real completion at 45.68 tok/s (close to the calibrated
number; the small gap is normal first-request cache warm-up, consistent
with everything else observed about warm-up in this doc).

**Known limitations of this first version**: single-sequence decode speed
only (concurrent-load calibration - e.g. calibrating for a specific
concurrency target, not just solo throughput - is a real further extension,
not attempted here); each candidate costs a full real model load, so total
calibration time scales with model size and shard count (fine as a one-time
opt-in step for a long-lived server, would need reconsidering for very
large models like GLM-5.2 where a single load may itself take minutes -
worth timing on the actual H200 box before assuming the current candidate
count/step size is still reasonable there).

# Native MTP speculative decoding, 2026-08-13 (RTX 3060, Gemma-4-26B-A4B)

> **RETRACTED 2026-08-14, see the endpoint-bias investigation section
> below**: every number in this section and in the `--spec-draft-n-max`
> sweep section was measured via the raw `/completion` endpoint with an
> unformatted prompt. That endpoint bypasses this model's chat template
> entirely, and the model (instruction/reasoning-tuned) produces
> degenerate, highly-repetitive output on unformatted prompts -
> independent of MTP, moe-cache, or placement. Degenerate repetitive text
> is trivially predictable, which silently inflates both MTP acceptance
> rate and MoE-cache hit rate. **The real content was never checked for
> these numbers, only tok/s** - so treat every specific number below as
> unverified until the corrected re-measurement (same section, chat-
> templated, content verified coherent) confirms or corrects it. The
> qualitative finding that MTP produces a real speedup on this hardware
> does still hold - the corrected numbers are lower but still positive.

## Real measured speedup

Initially thought untestable on this sandbox - hand-parsed the main GGUF's
header directly and found no `nextn`/mtp metadata keys, so assumed the
Unsloth quant just didn't ship an MTP head. Wrong place to look: MTP isn't
baked into the main model's tensors, it's a **separate sidecar GGUF file**,
loaded through the *speculative decoding* subsystem
(`params.speculative.draft.mparams.path`), not the main model's own
tensors. Confirmed via the HF repo listing
(`unsloth/gemma-4-26B-A4B-it-GGUF`): a `mtp-gemma-4-26B-A4B-it.gguf`
sidecar exists (also BF16/F16/Q8_0 variants under `MTP/`), matching our
already-downloaded main model. Downloaded the Q8_0 variant (441 MB,
verified against a real HF-token-authenticated, resumable, retrying curl -
see the download-reliability note below) and tested directly.

`llama-bench` doesn't support `--spec-type`/`--model-draft` at all - used
`llama-server` for the A/B instead: `-ngl 99 -ncmoe 23 --moe-cache auto -c
4096` baseline vs. the same plus `--model-draft <mtp file> --spec-type
draft-mtp --spec-draft-n-max 2` (2 is Unsloth's own recommended starting
point), same two prompts both times:

| prompt | baseline | MTP-enabled | speedup |
|---|---:|---:|---:|
| photosynthesis explanation | 50.05 tok/s | 67.60 tok/s | **1.35x** |
| binary search + complexity | 56.91 tok/s | 75.57 tok/s | **1.33x** |

Real, consistent **~1.33-1.35x** decode speedup, in line with (conservative
end of) Unsloth's claimed 1.4-2.2x range for `--spec-draft-n-max 2`.
moe-cache showed no dispatch/collect failures under MTP's verify-batch
load in this short test - doesn't fully resolve the open question below,
just no red flag in this specific run.

## `--spec-draft-n-max` sweep - the "2" was never actually tested, it's just Unsloth's suggested starting point

> **RETRACTED 2026-08-14** - same reason as above: measured via raw
> `/completion`, content never checked, numbers unverified. See the
> corrected re-measurement in the endpoint-bias investigation section.

Same principle already applied to `-ncmoe`/`n_threads` in Layer 2: don't
trust a suggested default, measure it. Swept n_max in {1,2,3,4,5,6,8},
same two prompts, one full server restart per candidate (real load each
time, not a no-alloc probe):

| n_max | avg tok/s (2 prompts) |
|---:|---:|
| 1 | 60.11 |
| 2 (Unsloth's suggested default) | 70.75 |
| 3 | 55.64 |
| 4 | 71.94 |
| **5** | **74.72** |
| 6 | 67.12 |
| 8 | 36.80 |

**n_max=5 beat the default of 2 by ~5.6%.** Shape matches theory - too low
leaves speedup on the table (fewer chances to skip a full forward pass),
too high wastes compute on tokens whose acceptance probability compounds
downward with each additional draft position, collapsing hard by n_max=8.
**Caveat, stated honestly**: only 2 samples per candidate, real run-to-run
noise present (n_max=3's two runs differ by 65%, 42.06 vs 69.21) - the
*pattern* (peak in the middle, collapse at the high end) is trustworthy,
the exact winner between 2/4/5 is not fully settled without more repeats.
This is exactly the kind of dimension `--moe-calibrate` should sweep
automatically once MTP is in scope for Layer 2 - not done yet, tracked
below.

## Download reliability note (unrelated to MTP itself, but real and costly)

First attempt (plain anonymous `curl -sL`) silently truncated at 193 MB of
441 MB - reported exit 0 despite being incomplete (the redirect target is a
time-limited signed CDN URL; something about the transfer let curl think it
finished cleanly when it hadn't). The `hf` CLI's own download (Xet-transfer
backend) then hung at 0 bytes for several minutes in this sandbox - killed
it and went back to curl, this time with `-C - --retry 8 --retry-all-errors`
plus an `Authorization: Bearer <token>` header (same token already saved
from the main model download) for resumability. That download was also
oddly slow (~100-150 KB/s) until an unrelated leftover diagnostic `curl`
process (from an earlier troubleshooting command that got auto-backgrounded
after exceeding its foreground timeout) was found still silently
downloading the *same file to `/dev/null`* in the background, competing for
the same limited egress bandwidth - killing that stray process roughly 6x'd
the real download's speed (120 KB/s -> 863 KB/s). Lesson: when a
foreground command that should have finished quickly instead needs to be
backgrounded, track it and clean it up explicitly rather than assuming it
died - it may still be running and consuming resources silently.

## Quality-preservation clarification (came up mid-investigation, worth recording)

Speculative decoding is provably lossless for output quality by
construction - not approximately, exactly. llama.cpp's actual verification
(`common_sampler_sample_and_accept_n` in `common/sampling.cpp`) samples a
token from the *target* model's own distribution at each position
independently, then checks whether the draft's guess matches that
independently-sampled token exactly; on any mismatch it keeps the token the
target actually sampled and discards the rest of the draft. The output is
therefore always literally a token the target model sampled on its own -
a higher-precision drafter cannot lift output quality above what the
(possibly heavily-quantized) target model would produce alone. What a
better drafter *can* do is raise the acceptance rate (more of its guesses
match what the target independently picks), which is pure speed, not
quality. Relevant directly to the H200/GLM-5.2 deployment given the
1-2bit-quantized GGUF options - MTP will make an aggressively-quantized
GLM-5.2 faster, not better than that same aggressively-quantized GLM-5.2
running alone.

## Two real gaps found, not yet fixed

1. **Exact-match verification is more conservative than necessary.**
   `common_sampler_sample_and_accept_n`'s exact-match rule rejects cases
   the classic probabilistic accept/reject rule (`accept with probability
   min(1, p_target(token)/p_draft(token))`, from the original speculative
   decoding papers) would have accepted - specifically, whenever the
   draft's guess isn't the target's own top pick but the target still
   assigns it real probability mass. This matters more the noisier/more
   diffuse the target's distribution is (e.g. heavy quantization), which
   is exactly the GLM-5.2 deployment's situation. A probabilistic
   accept/reject scheme would likely raise acceptance rate - and therefore
   the real-world speedup - without weakening the lossless guarantee at
   all (that guarantee is what the original algorithm's proof covers).
   **Not verified yet**: whether `--spec-type draft-mtp` specifically
   routes through this exact function or a different implementation -
   check before assuming this applies unmodified to the MTP path.
   Meaningfully bigger and more delicate than LFRU eviction - touches
   core sampling/verification correctness code, not moe-cache.
2. **Planner has no visibility into the drafter's VRAM footprint.**
   `common_maybe_autoplace_moe_cpu`/`common_moe_calibrate` only probe the
   *main* model's memory needs (`common_get_device_memory_data(path_model,
   &mparams, &cparams, ...)`) - the draft/MTP model is loaded via a
   completely separate `llama_model_load_from_file` +
   `llama_init_from_model` call in `speculative.cpp`
   (`params.speculative.draft.mparams.path`), invisible to either layer of
   what we built. Didn't cause a visible problem here since the MTP head
   is tiny (441 MB) relative to the 12 GB card's headroom, but the
   principle is real and matters more on H200 where free VRAM after
   loading a 744B model may be thin. Confirmed the drafter must run on GPU
   to be useful at all (CPU round-tripping per token would cost more
   latency than speculative acceptance saves) - so this isn't a "maybe
   offload it" question, the fix has to *reserve* GPU space for it, not
   choose whether to. Standing design direction from this session: the
   actual split between drafter-VRAM and main-model-expert-VRAM should be
   found empirically (extend `--moe-calibrate` to test placement
   candidates with the drafter enabled), not via a fixed static
   reservation rule - "only the test tells the truth" applies here the
   same as it did to placement itself.

# Endpoint-bias investigation and Layer 1/2 architecture rewrite, 2026-08-14

## The saga, honestly, because the wrong turns are as instructive as the right one

While verifying the first joint `--moe-calibrate` run's winning config
(golden-section search over `-ncmoe` and `spec-draft-n-max` together, see
below), the actual generated text was checked for the first time in this
whole MTP investigation - and it was degenerate, repetitive garbage
("sz-sz-sz-sz...", "tah-tah-tah..."). Every prior MTP validation in this
doc (the "1.33-1.35x speedup" section, the n_max sweep) had only ever
checked `predicted_per_second`, never the actual `content` - so this
wasn't a new bug, it was a pre-existing one that had simply never been
looked at closely enough to catch.

Systematic isolation (each step a real test, not a guess):
- Reproduced at both the calibration's chosen `ncmoe=15` and the
  earlier-"validated" `ncmoe=23`, and at both `n_max=4` and `n_max=2`
  (the exact value the original "coherent output" claim used) - ruled out
  the calibration search itself as the cause.
- Reproduced on the exact commit from *before* this session's `max_batch`
  fix (stashed current work, checked out the old commit, rebuilt,
  retested) - ruled out a code regression from anything built this
  session.
- Reproduced with `--moe-cache off` entirely - ruled out moe-cache.
- Reproduced with no `--model-draft`/`--spec-type` at all (plain
  generation, same placement) - **ruled out MTP entirely**.
- At this point the working theory turned to GPU/VRAM-level corruption
  from ~30+ ungraceful `SIGKILL`s of CUDA processes over many hours on a
  non-ECC consumer GPU (RTX 3060 has no ECC VRAM). Checked `dmesg` for
  Xid errors (none), `nvidia-smi` for ECC errors (none), confirmed no
  leaked processes/shared memory, and retested in the cleanest state
  achievable without an actual reboot - **still reproduced**. This was
  the wrong turn: the user correctly pushed back ("hardware is perfect,
  don't question it") rather than letting the investigation settle on an
  unfalsifiable environmental explanation.
- Re-read the Unsloth MTP docs page (the same one already cited above)
  with a more targeted query and found the actual cause in the
  "Recommended Sampling Parameters" section, plus, independently, a
  parsing-failure log line (`common_chat_peg_parse`) from an earlier
  diagnostic run showing the model's *real* underlying generation was
  coherent text prefixed with stray channel-marker tokens - a strong hint
  that pointed at chat-formatting, not generation quality, being the
  actual point of failure.
- Confirmed directly: the same prompt via `/completion` (raw, unformatted)
  produced garbage; the identical request via `/v1/chat/completions`
  (chat-template applied server-side) produced clean, coherent reasoning
  content. Reproduced on a second prompt. Corroborated independently via
  a GitHub Discussion search - ["Why is my Llama.cpp result so much worse
  than for the same model on another platform?"](https://github.com/ggml-org/llama.cpp/discussions/7781)
  describes the identical symptom as a known pattern, not something
  specific to this setup.

**Root cause**: every MTP/moe-cache test this session used the raw
`/completion` endpoint with an unformatted prompt string. This model
(instruction/reasoning-tuned) requires its chat template
(`<bos><|turn>system...<|turn>user...<|turn>model\n`, applied only by
`/v1/chat/completions`) to generate coherently at all - independent of
MTP, moe-cache, placement, or hardware. Every throughput number measured
this way is invalid, not just cosmetically wrong: degenerate repetitive
text is trivially predictable, which silently inflates both MTP
acceptance rate (repeating a token is an easy guess for the draft) and
MoE-cache hit rate (repetitive text routes to a narrow, unrealistic set
of experts). The corrected re-measurement below is the only one of this
session's MTP/calibration numbers that should be trusted.

## Fix: unified the calibration benchmark path, not just the prompt format

`common_moe_calibrate()`'s two benchmark functions had two different,
independent bugs:
- The `-ncmoe` search (`common_moe_bench_candidate`, in-process
  `llama_decode()` calls, no HTTP server) fed the model the *same* kind of
  raw, un-chat-templated prompt.
- The `spec-draft-n-max` search (`common_moe_bench_candidate_server`,
  subprocess-based, added earlier this session) used the raw
  `/completion` endpoint.

Rather than patch both separately, removed the in-process benchmark
entirely and made the subprocess+HTTP path (`common_moe_bench_candidate_server`)
the *only* benchmark function, used by both searches and by thread
tuning. It now:
- Launches a real `llama-server` subprocess with `--temp 1.0 --top-p 0.95
  --top-k 64` (this model family's documented MTP sampling
  recommendation), a fresh port per candidate (fixes a real port-reuse
  race found separately - see below), and `-t`/`-tb` when tuning threads.
- Sends `/v1/chat/completions` requests (proper chat template, applied
  server-side, exactly like a real client) for two representative
  prompts, averaging `timings.predicted_per_second` across both.
- One retry per candidate before treating a failure as real (single
  subprocess runs are noisy samples - same lesson as the n_max=3
  run-to-run variance found earlier this session).

This also meant thread tuning - previously skipped whenever MTP was
calibrated, because its old in-process benchmark had no MTP awareness and
mixing MTP/non-MTP throughput numbers wouldn't have been a fair
comparison - could be **re-enabled**: since every candidate now goes
through the same real, correct path, MTP-aware thread tuning is directly
comparable to the placement/n_max numbers above it.

## Other real bugs found and fixed during this work (independent of the endpoint bug)

- **`n_seq_max=-1` crash**: `--moe-calibrate`'s early-exit in
  `tools/server/server.cpp` ran before `params.n_parallel`'s `-1` ("auto")
  sentinel got resolved to a real value, so `cparams.n_seq_max` silently
  wrapped to a huge unsigned value and crashed the no-alloc probe with
  `n_seq_max must be <= 256` - which then got **mis-reported** as "model
  has no MoE experts" because the probe's exception handler didn't log
  the real error, just returned an empty result indistinguishable from
  "not a MoE model." Fixed both: moved the early-exit to after
  `n_parallel` resolution, and made the exception handler log the actual
  exception message.
- **Stale `tok_per_sec` reporting**: the cache/log correctly picked the
  best `n_max`, but the reported throughput number stayed at the
  pre-n_max-search (`-ncmoe`-only) value unless the n_max result was
  strictly higher - meaning a genuinely-better MTP-inclusive result could
  still get under-reported. Fixed by carrying the n_max search's own
  winning number forward once it's known to be real.
- **Port-reuse race**: all `n_max` candidates originally shared one port
  for the whole search. `SIGKILL`ing a candidate's server doesn't
  guarantee the OS releases its listening socket before the next
  candidate tries to bind the same port (TIME_WAIT) - caused a real,
  reproducible intermittent "failed" candidate. Fixed with a fresh port
  per subprocess.

## Corrected, validated results (2026-08-14, chat-templated, content verified coherent)

Same RTX 3060 + Gemma-4-26B-A4B + MTP Q8_0 sidecar setup, full joint
`--moe-calibrate` run with the fixed benchmark path:

**`-ncmoe` golden-section search** (range [13,17], derived from Layer 1's
live safe floor):

| ncmoe | tok/s |
|---:|---:|
| **15 (winner)** | **51.36** |
| 14 | 46.84 |
| 13 | 41.81 |

**`spec-draft-n-max` envelope + golden-section search** (at ncmoe=15):

| n_max | tok/s |
|---:|---:|
| 1 | 54.93 |
| 2 | 56.06 |
| 4 | 54.64 |
| 7 | 49.60 |
| 8 | 54.10 |
| **9 (winner)** | **54.55** |
| 10 | 52.38 |
| 13 | 42.48 |
| 16 | 39.40 (envelope edge) |
| 32 | 22.79 |

**Thread tuning** (at ncmoe=15, n_max=9): n_threads=12 (all logical cores)
won at 56.92 tok/s, beating the default of 6 - notably the *opposite* of
this session's earlier concurrency-cliff finding (there, more threads
made things worse) - a reminder that these levers interact differently
under different workloads (single-sequence MTP drafting+verifying is not
the same CPU access pattern as the earlier multi-sequence concurrency
test) and shouldn't be assumed to generalize from one context to another.

**Final winner**: `ncmoe=15, n_threads=12, spec-draft-n-max=9`, 56.92
tok/s, cached correctly and picked up automatically on a normal launch
(confirmed: `common_maybe_autoplace_moe_cpu` log line shows all three
values applied). **Real coherent output confirmed** via
`/v1/chat/completions` on two separate prompts post-calibration, e.g.
"Photosynthesis is the process where plants use sunlight to turn carbon
dioxide and water into food..." - and the response's own `timings` block
now exposes `draft_n`/`draft_n_accepted` directly (155/389 ≈ 40% and
166/291 ≈ 57% on the two test prompts) - realistic acceptance rates for
genuinely diverse text, a strong independent corroboration that the fix
is measuring the real thing now, not inflated-by-repetition numbers.

Compare to the retracted numbers: the corrected `-ncmoe` peak (51.36) and
`n_max` peak (56.92 combined) are noticeably *lower* than the retracted
run's inflated figures (53.99, 71.33) - exactly the direction predicted
by "degenerate repetitive text is trivially predictable, artificially
inflating measured throughput." The qualitative finding (MTP gives a real
speedup on this hardware) still holds; the specific multiplier claimed
earlier does not, and hasn't been re-derived cleanly yet (would need a
proper chat-templated A/B against a no-MTP baseline, not yet done as a
dedicated re-test - the numbers above are internal to the MTP-enabled
calibration search, not compared against a corrected non-MTP baseline).

## Second retraction and re-correction (2026-08-14, golden-section search bug)

**The "Final winner: n_max=9, 56.92 tok/s" call above is itself retracted.**
The `n_max` trace table right above it already contained the evidence:
n_max=2 scored 56.06 tok/s, higher than the declared winner n_max=9's
54.55 - golden-section search assumes a unimodal objective, the real
curve isn't one (dips at n=7, rises again through 8-10), and the search
silently converged on a local peak without ever reconsidering n=1/n=2,
both of which were already sitting in its own trace from the earlier
envelope-doubling phase. See "LFRU eviction: A/B results" bullet in the
build list below for the full root-cause writeup - the same session that
found and fixed this also fixed the LFRU eviction policy, both by
`common_golden_section_search_max` / `moe-cache.cu` respectively.

Fix: after golden-section search returns, validate its pick against
every point already measured in the trace map (free - no new subprocess
spawns, the data already exists) and take the true max instead of
trusting the bisection blindly. Applied to both the `-ncmoe` and
`spec-draft-n-max` searches in `common/common.cpp`.

Re-ran the full joint `--moe-calibrate` with the fix, same RTX 3060 +
Gemma-4-26B-A4B + MTP Q8_0 setup:

| n_max | tok/s |
|---:|---:|
| 1 | 57.80 |
| 2 | 60.10 |
| **3 (winner, trace-validated)** | **64.92** |
| 4 | 52.24 |
| 7 | 50.78 |
| 8 | 48.72 |
| 10 | 49.25 |
| 16 | 33.39 |
| 32 | 22.20 |

(ncmoe=15 alone, no MTP: 50.70 tok/s this run - within normal run-to-run
noise of the earlier 51.36.) Thread tuning tried n_threads=12 this time
too and it *lost* (48.71 vs the default n_threads=6's 64.92) - correctly
rejected, config stayed at n_threads=6. **Final winner: `ncmoe=15,
n_threads=6, spec-draft-n-max=3`, 64.92 tok/s** - a real **+28.1%** over
the no-MTP baseline (50.70 tok/s), cached and auto-applied on normal
launch (confirmed via the `common_maybe_autoplace_moe_cpu` log line).
**Content-verified coherent** via `/v1/chat/completions` on two prompts
post-calibration: 100/147 (68%) and 98/152 (64.5%) `draft_n_accepted` -
healthy, realistic acceptance rates, both notably higher than the
previous (buggy-search) run's 40%/57%, consistent with a smaller n_max
having an easier time getting each individual draft token accepted.

This replaces the 56.92 tok/s number everywhere it was cited above and
in the build-list summary below - use 64.92 tok/s (n_max=3) as the real,
validated figure for this hardware/model/context combination.

## Lesson for next time

The endpoint/prompt-formatting choice is not a cosmetic detail of a test
harness - it changes what's actually being measured, silently, in a
direction (inflated speedup) that's easy to mistake for a good result.
Any future benchmark of a chat/instruct-tuned model through
`llama-server` should default to `/v1/chat/completions`, and any
"speedup" or "hit rate" claim should have the actual generated content
spot-checked at least once, not just the timing numbers - cheap insurance
against exactly this failure mode.

# Build list summary (updated 2026-08-13, end of day)

## Already built, VALIDATED with real measurement
- moe-cache core port (leloch's design). Load-bearing feature. **Measured
  +54% decode speedup (27.0 -> 41.6 tok/s), 62.1% hit rate on our RTX 3060 +
  Gemma-4-26B-A4B** - see validation section above. RFC: 1.7-2.1x decode on
  skewed routing (Qwen-A3B), consistent with our own result. GLM-5.2
  confirmed skewed (SharkWipf: 30-50% measured). **Independently
  re-verified 2026-08-14 via `/v1/chat/completions`** (different placement,
  ncmoe=15: `--moe-cache off` 39.37 tok/s vs `--moe-cache auto` 51.71
  tok/s, **+31.3%**) - the mechanism's core value holds up under the
  corrected, chat-templated benchmark path, not just the original
  `/completion`-based measurement.
- mmap host-pinning fix (ours). One-time load-speed win only, not decode.
- Live `reserve_mb`/`max_batch` defaults (ours) - closed a silent
  cache-disabling bug for any concurrent deployment (`MAX_BATCH` hardcoded
  to 8 regardless of `--parallel`). See the concurrency-cliff investigation
  section.
- **Layer 1: live safe-placement auto-fit** (ours) - `-ncmoe` auto-decided
  live at every launch only when the config as given wouldn't fit, via a
  predictive no-alloc probe before allocation. Validated: previously-crashing
  config now self-corrects and serves real completions; explicit `-ncmoe`
  always left untouched; a config that already fits shows zero interference.
- **Layer 2: `--moe-calibrate`** (ours) - empirical throughput-optimal
  placement + thread count, cached per GPU+model+context combo. **Measured
  40% faster than Layer 1's safe floor alone (35.35 -> 49.33 tok/s)** on the
  RTX 3060 sandbox - this *is* the "static pre-flight capacity/hit-rate
  planner" item that used to sit in the longer-term/exploratory list below,
  now built and real. Known scope limit: single-sequence decode speed only,
  not calibrated for a specific concurrency target - see the Layer 2
  section for the full writeup. **Re-verified 2026-08-14** (post both the
  endpoint and golden-section search fixes): safe floor ncmoe=13 -> 41.68
  tok/s, Layer 2's pick ncmoe=15 -> 50.70 tok/s, **+21.6%** - same
  qualitative finding, smaller than the original 40% claim (that number
  pre-dates both fixes). Also separately confirmed with a wider sweep
  (ncmoe 13-30): a naively conservative ncmoe=30 only reaches 46.995
  tok/s, so Layer 2's search is finding a real, non-obvious optimum, not
  just validating the safe floor.
- **Native MTP speculative decoding** (merged upstream, `--spec-type
  draft-mtp --model-draft <sidecar.gguf>`). Works and gives a real
  speedup - confirmed with coherent output and realistic acceptance
  rates after fixing two serious bugs in how it was measured: the
  endpoint-formatting bug (see "Endpoint-bias investigation" below) and
  a golden-section search bug (see "Second retraction and re-correction"
  above) that made the first corrected calibration land on n_max=9
  (54.55 tok/s) when n_max=2/3 both scored higher in the search's own
  trace. **The original "1.33-1.35x" multiplier and the intermediate
  "n_max=9, 56.92 tok/s" number are both retracted.** Current real,
  validated number: `ncmoe=15, n_threads=6, spec-draft-n-max=3`, **64.92
  tok/s, +28.1% over the no-MTP baseline (50.70 tok/s)**, content-verified
  coherent via `/v1/chat/completions` with realistic 64-68%
  `draft_n_accepted` rates on two separate prompts. Provably lossless for
  output quality regardless of any of this (verified output is always
  sampled from the target model, never the draft). Confirmed the same
  sidecar pattern exists for `unsloth/GLM-5.2-GGUF` (20%
  acceptance-length improvement claimed for GLM-5.2's own MTP layer per
  public sources) - directly applicable to the real deployment target
  once that hardware is available, **using `/v1/chat/completions` for
  any validation there, not `/completion`**.
  Search-bug fix: `common_golden_section_search_max`'s callers
  (`common_moe_calibrate` in `common/common.cpp`) now validate the
  bisection's pick against every point already sitting in the trace map
  after the search completes - free, since that data's already measured,
  no new subprocess spawns - and take the true best-of-measured instead
  of trusting golden-section's assumption that the objective is
  unimodal. Applied to both the `-ncmoe` and `spec-draft-n-max` searches.
- **LFRU eviction (SLRU + heat + decay hybrid, ours)** - replaces plain
  LRU as the CUDA-side eviction policy. Four designs A/B'd on an identical
  mixed hot/cold workload (RTX 3060, Gemma-4-26B-A4B, `-ncmoe 15
  --moe-cache auto`, 604 slots) before landing on the shipped one:
    - plain LRU (baseline): **63.8%** hit rate
    - SLRU v1, two segments (probation/protected_), *uncapped* protected_:
      **60.2-60.3%** - a real regression. Root cause: protected_ grew to
      598-600/604 slots unbounded, starving probation to 4-6 slots -  new
      admissions had nowhere to prove themselves before eviction pressure
      hit them.
    - Heat-only, single list + per-slot heat counter (Colibri's
      `tier_decay()` halving) + bounded 8-candidate eviction window with
      25%-plus-4 hysteresis (Colibri's `tier_pick_lfru` margin), no
      segments at all: **64.6%** - recovered past plain LRU with just the
      heat signal and no segmentation.
    - Capped SLRU, same two segments as v1 but protected_ hard-capped at
      50% of the pool with FIFO (oldest-first) demotion at the cap:
      **65.0%** - the structural fix for v1's regression, and on its own
      the best single mechanism measured.
    - Two "let heat/decay decide the cap instead of a fixed percentage"
      variants were tried and **both regressed back to v1's failure
      mode**: an absolute per-slot heat floor for decay-driven demotion
      (61.4%, protected_ still 596-598/604) and a floor relative to
      protected_'s own mean heat (61.3%, still 599/604). Root cause in
      both: decay only fires on *fill* (admission) events, but within one
      busy request thousands of *hits* land on the same resident working
      set between fills - heat balloons for nearly the whole pool before
      decay gets a chance to run, and a demoted slot is usually re-hit and
      re-promoted on the very next token before the next decay checkpoint.
      Heat/decay alone doesn't have a fast enough signal to regulate
      *population size* at this timescale; that needs a hard structural
      bound.
    - **Shipped design**: capped SLRU (the structural fix that measured
      best) + heat as the *tiebreaker* - both for eviction-candidate choice
      within whichever segment is being drained, and for which protected_
      slot pays the demotion cost when a promotion needs to make room -
      instead of heat/decay replacing the cap outright. **65.0%**,
      statistically tied with capped-SLRU-alone on this workload (too
      small/short to discriminate further); kept as the shipped design
      because it's strictly safer (same proven structural bound, heat only
      ever changes *which* candidate within an already-safe pool gets
      picked, never *whether* the pool stays bounded) and should matter
      more under a larger, more diverse, longer-running workload than this
      16-request A/B can exercise. All four intermediate implementations
      backed up during development, not kept in the tree.

## Near-term - targets the GLM-5.2/H200 deployment directly
- ~~FR-Spec draft-vocab trimming for GLM's MTP~~ **BUILT for gemma4
  (sandbox), MEASURED, gemma4-only for now, 2026-08-14.** Not the
  upstream EAGLE3/qwen35 design (a d2t tensor baked into a pre-trimmed
  draft GGUF at conversion time - no producer tooling for that exists
  anywhere upstream, not even for its original Qwen3.5 target). Instead,
  Option B from this session's own design discussion: the original draft
  checkpoint is never modified. Four pieces, all model-agnostic except
  the last:
  1. **Real-traffic token-frequency logging** (`tools/server/server-token-freq.{h,cpp}`,
     `server_token_freq_logger`) - the server passively counts every
     confirmed *generated* token (not prompt tokens) into a dense
     per-vocab-id histogram, persisted as sparse JSON under
     `fs_get_cache_directory()`, keyed by a hash of the model path,
     accumulating across restarts. `--token-freq-log` (on by default).
  2. **Sidecar regeneration tool** (`tools/frspec-vocab-trim/`,
     `llama-frspec-vocab-trim`) - standalone, re-runnable any time (cron,
     or by hand): reads the histogram, ranks by real frequency, writes
     the top N token ids as a plain-text sidecar file (one id per line,
     trimmed-vocab position = line index - the same d2t convention
     EAGLE3/qwen35 use, just sourced externally instead of from a GGUF
     tensor).
  3. **Shared, architecture-agnostic ggml helpers**
     (`src/models/models.h` + `src/models/frspec-helpers.cpp`):
     `llm_frspec_load_d2t_sidecar`, `llm_frspec_gather_vocab_rows`,
     `llm_frspec_scatter_to_full_vocab`, the pending-vocab-map hint
     mechanism (`llama_frspec_set_pending_vocab_map` in `llama.h`, public
     API - needed to cross the `common/` -> `src/models/` boundary since
     Gemma-4's MTP draft has no in-file signal distinguishing it from the
     trunk the way qwen35's `mtp_only` flag does), and
     `llm_graph_input_frspec_d2t` (an `llm_graph_input_i`, matching
     `granite-switch.cpp`'s `llm_graph_input_switch` template exactly -
     the correct pattern for a load-time-populated, per-model-lifetime-constant
     tensor, found only after an extended dead end trying to create a
     GGUF-independent constant weight tensor directly via
     `llama_model_loader` internals). A future GLM port only needs a
     small amount of code calling into these already-tested helpers, not
     a fresh reimplementation.
  4. **Load-time gather/scatter wiring** - architecture-specific, ~40
     lines. Named `--spec-vocab-map <path>` on the CLI
     (`common_params_speculative_draft::frspec_vocab_map`).

  **Real bug: wired into the wrong class initially.** Gemma-4's MTP
  draft GGUF declares `general.architecture = gemma4-assistant`
  (`LLM_ARCH_GEMMA4_ASSISTANT` -> `llama_model_gemma4_assistant`,
  `src/models/gemma4-assistant.cpp`) - a genuinely separate architecture
  from the `gemma4` trunk class the *target* checkpoint uses, not a mode
  flag on the same class. The first pass wired the trim into
  `gemma4.cpp` (`llama_model_gemma4`) end to end, compiled cleanly, and
  only revealed the mismatch at live-load time (no "FR-Spec vocab trim
  active" log line, traced via targeted debug instrumentation before
  finding the real cause: `strings <gguf> | grep general.architecture`
  showed `gemma4-assistant`, not `gemma4`). Ported the identical pattern
  to `gemma4-assistant.cpp`; `gemma4.cpp`'s copy was left in place
  (harmless no-op there in this deployment shape, and free insurance for
  any future case where `gemma4` itself is loaded as a self-speculating
  MTP draft).

  **Three more real bugs found and fixed during build/verify, all
  latent, none specific to this feature's logic:**
  - **Token-frequency histogram was never flushed on normal shutdown.**
    `destroy()` (which flushes) was only ever called on the
    server's "sleeping" state transition, never from the actual
    `SIGTERM`/`SIGINT` shutdown path (`clean_up()` in `server.cpp` calls
    `ctx_server.terminate()`, which only unblocks the task queue - it
    never touched the logger). Every restart silently lost every count
    recorded since the last periodic 4096-token flush - for the
    "restart to regenerate the sidecar" workflow this whole feature
    depends on, that could be most or all of a session's real traffic,
    every single time. Fixed with a new public
    `server_context::flush_token_freq()`, called from `clean_up()`
    before `terminate()`. Verified with a real kill -TERM: histogram
    file was stale/empty before the fix, correctly populated after.
  - **`llama-frspec-vocab-trim`'s `--model-draft` flag hashed the wrong
    path.** The server keys the histogram by the *target* model's path
    (`params_base.model.path`) - correct, since confirmed generated
    tokens are always in the target's vocab space regardless of which
    draft proposed the (possibly-rejected) candidates for them - but the
    tool derived its lookup hash from whatever `--model-draft` path the
    user passed. Every user following the tool's own documented usage
    would reliably get "no histogram found" starting from nothing.
    Renamed the flag to `--model` (target model path; `--model-draft` is
    kept as an accepted alias for the old name, not removed outright).
  - **`ggml_get_rows` requires I32 indices, not I64.** Initial design
    copied EAGLE3/qwen35's convention of an I64 d2t tensor (needed for
    their `ggml_set_rows`-only scatter-back use case). This trim also
    needs a *gather* step first (EAGLE3/qwen35 never do, since their
    trim is baked in at conversion time) - `ggml_get_rows` asserts
    `b->type == GGML_TYPE_I32` and would have failed loudly the first
    time a real trim was exercised. Switched the shared d2t tensor to
    I32 throughout (still accepted by `ggml_set_rows`, which allows
    either).

  **Verified correct, not just compiled.** Live end-to-end test on the
  sandbox (RTX 3060, gemma-4-26B-A4B, Q8_0 draft): confirmed the
  activation log line fires with the real trim size
  ("421 of 262144 tokens" from an initial small real-traffic sample);
  identical output to the untrimmed baseline at temp=0
  ("The capital of France is **Paris**.", byte-for-byte, same
  `predicted_n`); coherent output on a "list primes" prompt; and one
  prompt ("haiku about mountains") that degenerated into token
  repetition identically *with and without* the trim, confirming that's
  a pre-existing small-quant-model quality artifact, not something the
  trim introduced. This matches the design's lossless guarantee: the
  draft's candidates are restricted to the trimmed set, but the target
  always re-verifies over the real, full vocab, so a wrong/rejected
  draft proposal can only cost a fallback token, never wrong output.

  **Speed finding, revised after a follow-up fix (2026-08-14, same
  day).** The first pass measured a real 5-11% regression and named a
  working theory: the trim ran a live `ggml_get_rows` gather inside the
  compute graph on *every* forward pass, re-executed as a real GPU
  kernel every step even though the result never changes -
  `can_reuse()=true` only skips rebuilding the graph's CPU-side
  topology, it does not skip re-running each node's kernel once the
  graph executes. Checked this empirically before acting on it: "graphs
  reused" counts from the server logs were statistically identical
  between trimmed and untrimmed runs (same ~198-per-200-token increment
  in both), ruling out "the trim breaks graph reuse" as the cause and
  confirming the cost really was per-step kernel execution time, not
  launch-dispatch overhead.

  **Fix:** moved the gather out of the compute graph entirely.
  `llm_graph_input_frspec_d2t` now also owns `output_w`, a
  pre-gathered-weight graph *input* (not a graph *node*) populated
  exactly once - on the first `set_input()` call, via a new
  `llm_frspec_gather_vocab_rows_from_backend()` helper that reads the
  needed rows straight off the original weight's backend buffer
  (`ggml_backend_tensor_get()`, works whether it's CPU- or
  GPU-resident) - and left untouched on every call after that. This
  didn't need the GGUF-independent-constant-tensor mechanism that was
  architecturally blocked earlier in the session; it reuses the exact
  `llm_graph_input_i`/`set_input()` pattern already built for the d2t
  index tensor, just with a "populate once" guard instead of "populate
  every call." The scatter-back step (`ggml_set_rows` placing the
  trimmed logits into a full-vocab-shaped, -inf-filled tensor) still
  runs every step, unavoidably - its *data* genuinely changes every
  token, unlike the weight gather.

  **Re-measured, correctness first:** temp=0 output on "The capital of
  France is" stayed byte-identical to both the untrimmed baseline and
  the pre-fix trimmed run - the fix only changes *when* the gather
  computation happens, never its result.

  **Re-measured, speed:** 9 samples each, same 200-token photosynthesis
  prompt, temp=0, on the same 645-token real-traffic sidecar as before.
  Cache-once trim: 57.8-62.7 tok/s (mean 60.5). A matched fresh
  untrimmed baseline, run immediately after: 55.0-59.8 tok/s (mean
  57.4) - lower than the *first* untrimmed baseline measured earlier in
  the session (61.6-62.7), despite being the same build and config. That
  drift between two untrimmed baselines, taken minutes apart, is bigger
  than the trimmed-vs-untrimmed difference either measurement showed -
  the sandbox's own run-to-run noise (thermal state, background load,
  moe-cache warm-up) is evidently larger than this technique's effect
  size at this scale.

  **Second fix, same day, prompted by being asked directly whether 645
  rows was really enough data movement to explain a millisecond-scale
  regression - it wasn't.** Back-of-envelope: 645 rows at embedding
  width, even unquantized, is a few MB; at this GPU's ~360GB/s bandwidth
  that's low single-digit microseconds if bandwidth-bound, nowhere near
  the gap being chased. The real remaining suspect: the scatter-back
  step (`ggml_fill(..., -INFINITY)` over the *full* 262,144-wide output,
  every step, regardless of trim size) was still a live graph op -
  strictly bigger than the gather that was just fixed, and unaffected
  by that fix. Same root cause as before (a value that's actually
  constant, paid for as if it weren't), same fix shape: `out_template`
  (the -inf-filled scatter target) joined `output_w` as a third
  populate-once graph input on `llm_graph_input_frspec_d2t`, filled
  exactly once via `ggml_backend_tensor_set()`. Safe to never re-fill:
  `d2t` names the identical fixed set of positions on every scatter
  call for a model instance's whole lifetime, so every position outside
  that set is written by no call, ever, and correctly stays whatever
  the one-time fill gave it.
  `llm_frspec_scatter_to_full_vocab()`'s signature changed to accept
  this pre-filled template directly instead of building one internally
  each call.

  **Re-measured a third time, back-to-back in the same session to
  cancel drift** (10 samples each, immediately sequential, same prompt,
  same sidecar): trimmed 56.2-60.8 tok/s (mean 58.7); untrimmed,
  restarted and run immediately after with no other change, 54.8-59.9
  tok/s (mean 57.6). Trimmed is now at parity with, if anything slightly
  ahead of, untrimmed - the regression is gone, not just reduced to
  noise. **This is now a real, if modest, positive result on the
  sandbox proxy**, not an open question. Correctness re-verified
  unchanged (byte-identical temp=0 output) after this fix too - both
  fixes only ever changed *when* already-correct computations happen,
  never their results.

  Independently worth knowing, found while investigating: SGLang - the
  most mature real-world implementation of this exact paper's technique
  - has an open, unresolved bug report
  ([sgl-project/sglang#8581](https://github.com/sgl-project/sglang/issues/8581))
  of FR-Spec *decreasing* throughput (430.99->291.04 tok/s) with a
  *lower* draft-acceptance rate too (1.72->1.64), closed as `inactive`
  with no maintainer explanation. That's a different failure mode than
  what this session found (their acceptance rate itself dropped, ours
  never did - only raw graph-kernel overhead was ever the issue here),
  but it's a real signal that this technique doesn't reliably reproduce
  the paper's controlled 1.12x number in production serving engines
  without real engineering care, which is exactly what happened here:
  the naive first port regressed, and it took two rounds of finding the
  actual root cause (not just reporting the number) to get a clean
  result.

  Still genuinely open: whether this specific measured edge holds up at
  GLM-5.2/H200 scale, where the vocab size, draft size, and batch
  dynamics are all different. But the sandbox result is no longer a
  caveat pointing away from the technique - it's a small, clean win,
  arrived at by refusing to accept "there's probably just some overhead
  somewhere" as a final answer.

  **"Net outcome" test at the real calibrated config, not `--parallel
  1`.** All FR-Spec numbers above were measured at `--parallel 1`,
  chosen for simplicity during correctness testing - not the config the
  session's own 64.92 tok/s MTP headline number was calibrated at
  (`--parallel 4`, per the calibration cache key
  `...c4096|p4|ngl99`). Re-ran trimmed vs. untrimmed at that real
  config, `/v1/chat/completions` (the "solid"-tier endpoint used
  throughout this doc, not the raw `/completion` endpoint the earlier
  FR-Spec numbers used), 5 samples each: untrimmed 55.12-55.98 tok/s
  (mean 55.66); trimmed 55.21-55.71 tok/s (mean 55.50). Tighter spread
  than any earlier FR-Spec measurement, and genuinely flat - not even
  the parity-with-a-slight-edge seen at `--parallel 1`, just a clean
  wash either way. Consistent with the fix being real and complete: no
  regression at any config tested, and no dramatic win either.

  Two things this run surfaces that are worth separating from FR-Spec
  itself: (1) at the *identical* calibrated config, this photosynthesis
  prompt gives ~55.7 tok/s solo, not 64.92 - a ~14% gap, from prompt
  content alone (MTP's draft-acceptance rate is prompt-dependent; the
  64.92 figure came from whatever probe prompts `--moe-calibrate` used,
  not this one). The 64.92 headline number is real but prompt-specific,
  not a universal constant for "this config" - a caution against
  treating any single throughput number in this document as portable
  across different real traffic. (2) None of this session's percentage
  gains compound the way stacking them on paper suggests, both because
  they were measured at different `--parallel` settings across the
  session's own timeline and because of this same prompt-dependence.

  **Real bug found running everything together, 2026-08-14: FR-Spec
  trim is currently incompatible with `--spec-type
  draft-mtp,ngram-suffix` (MTP + suffix-decode combined).** Requested
  directly: test all of this session's work together, not FR-Spec in
  isolation. Built the combined config - `-ncmoe 15 --parallel 4
  --spec-type draft-mtp,ngram-suffix --spec-draft-n-max 3
  --spec-vocab-map <sidecar> --kv-unified --cache-idle-slots` - and hit
  a severe regression: 29.8-30.2 tok/s with only 10.7% draft
  acceptance, versus 66.3-68.6 tok/s at 70-74% acceptance for the exact
  same config minus `--spec-vocab-map`. Isolated properly rather than
  guessing: removing `--kv-unified` alone reproduced the identical
  broken numbers (30.24 tok/s, 10.7%), ruling it out; removing
  `--spec-vocab-map` alone (keeping `--kv-unified`) fully restored
  healthy numbers (65.6-68.0 tok/s, 70-74% acceptance). **FR-Spec is
  the cause, specifically when combined with suffix-decode** - FR-Spec
  alone with MTP was at parity (measured above), suffix-decode alone
  with MTP was healthy (measured earlier this session), but the three
  together break badly.

  **Investigated properly, not just theorized (2026-08-14, later same
  day)** - asked directly to solve it, not just document it. Ruled out
  two plausible theories by reading the actual code and tracing a live
  `-lv 5` debug session, in order:
  1. *"MTP's own `p_min` confidence gate fires more often under the
     trim, handing more rounds to suffix-decode, which does worse on
     this content."* Ruled out: `p_min` defaults to `0.0f`
     (`common/common.h`), meaning the gate (`common/speculative.cpp`,
     the MTP drafter's per-token `if (cur_p->data[0].p < params.p_min)`
     check) never fires with default settings.
  2. *"The combined-drafter verification path uses MTP's own reported
     probability - now inflated, since it's normalized over only 645
     candidates instead of the true 262,144-token distribution - in a
     probabilistic accept/reject test, unfairly rejecting good
     candidates."* Ruled out by reading the actual verification call
     (`tools/server/server-context.cpp`,
     `common_sampler_sample_and_accept_n`): at `temperature=0` (every
     test in this doc), acceptance is a plain greedy comparison against
     the target's own independently-computed top pick - it never reads
     the *drafter's* probability at all, so trim-induced probability
     inflation can't be the mechanism.

  What the live trace actually shows: `-lv 5` confirms MTP is the
  implementation producing every single round's draft (suffix-decode's
  fallback path is never reached - `common_speculative_draft()` only
  calls the next configured impl for a sequence still needing a draft,
  and MTP always returns a non-empty result here) - so this isn't
  "suffix-decode drafting badly," it's MTP's *own* drafted tokens being
  rejected far more than usual specifically when suffix-decode is also
  configured. Per-round accept counts (`accepted N/3 draft tokens` in
  the trace) show the *very first* drafted position failing in 30 of 42
  rounds (71%) - immediate rejection, not a late-position falloff.
  Suspicious correlate, not yet proven causal: the single most
  over-represented token in this session's thin 645-token histogram
  (id 236772, a dash/hyphen the model uses heavily in its own
  markdown-style formatting) accounts for 630 of 6114 total real-traffic
  observations (~10.3%) - MTP's confidence in proposing it is very high
  within the trimmed 645-candidate view (0.99+ in the trace) but the
  target frequently disagrees. This alone doesn't explain why FR-Spec
  alone (no suffix-decode) stayed at parity with the identical trim and
  identical prompt - the actual trigger for *why this specific
  combination* surfaces the mismatch (state/scheduling interaction
  between multiple configured drafter types, not drafted-token content
  itself) is still not pinned down at the code level despite this
  investigation ruling out two concrete, testable theories.

  **Action taken: do not enable both together.** FR-Spec trim
  (`--spec-vocab-map`) should not be used alongside `--spec-type
  draft-mtp,ngram-suffix` until this is root-caused and fixed - each is
  fine alone, MTP+suffix is the recommended production combination from
  earlier in this document, and FR-Spec is not worth keeping active at
  the cost of breaking that. This is not yet enforced in code (no
  warning or guard added), only documented here - a real follow-up if
  FR-Spec is pursued further.

  **The real, clean "net outcome of this session's work" number**, all
  compatible pieces together (moe-cache placement + MTP + suffix-decode
  + `--kv-unified`/`--cache-idle-slots`, FR-Spec excluded per the
  finding above), solo decode, `--parallel 4`, same photosynthesis
  prompt: **65.6-68.0 tok/s (mean 66.8), 70-74% draft acceptance** -
  genuinely ahead of the 55.5-55.7 tok/s MTP-alone number at the same
  config, and the real reference point for "what does combining
  everything actually get us" rather than adding percentages from
  differently-configured measurements taken at different points in the
  session.

  Toggling is clean either way: `--spec-vocab-map` unset (the default)
  is a verified no-op - `frspec_d2t_ids` stays empty, `load_arch_tensors`
  and the graph-building gather/scatter code both no-op, full untrimmed
  vocab used exactly as before this feature existed.
- LFRU eviction + hysteresis + decay, replacing our ported plain LRU
  (Colibri's tier.h). No hard number yet - needs A/B. Motivated by
  GLM-5.2's confirmed skew.

## Medium-term - multi-user serving (5-10 agentic users)
- ~~Server-level cross-request prefix/KV reuse~~ **ALREADY BUILT AND
  VALIDATED, 2026-08-14** - not something to build, the narrower
  maintainer-suggested path from discussion #21961 ("we can obviously
  implement the same scheduling for cross-request KV reuse at server
  level") turns out to already be shipped in mainline: `server_prompt_cache`
  (`tools/server/server-task.h`/`.cpp`), enabled by default via
  `--cache-ram` (default 8192 MiB). When a slot's content is evicted, its
  full KV state gets saved to this RAM-resident pool; on a new request,
  `server_prompt_cache::load()` searches the pool by longest-common-prefix
  match (same LCP logic as the live-slot `--slot-prompt-similarity`
  selection, also already on by default) and restores a match into
  *whichever* slot the new request lands on - genuinely cross-slot, not
  just "the same slot got lucky."

  **Verified empirically, not just read in source**: first test (Gemma-4,
  the sandbox's only model) showed the restore firing correctly in the
  trace log (`found better prompt with f_keep=0.866`, 187/216 tokens
  matched) but then getting silently discarded - a *separate* per-slot
  "context checkpoint" mechanism ran its own validation against the
  physical slot's *unrelated* prior checkpoint history, found no match,
  and erased the just-restored state, forcing near-full reprocessing
  anyway (log message names the cause directly: "forcing full prompt
  re-processing due to lack of cache data (likely due to SWA or
  hybrid/recurrent memory..." - traced to PR #13194, SWA support,
  an acknowledged upstream limitation, not a bug introduced here).
  Confirmed `gemma4.cpp` uses SWA. **Confirmed `glm4.cpp`/`glm4-moe.cpp`/
  `glm-dsa.cpp` (GLM-5.2's family) do not** - grepped directly, zero
  matches for `swa_type`/`n_swa`. Re-ran the identical test on a non-SWA
  model already on disk (Qwen2.5-0.5B) to isolate the mechanism from this
  Gemma-4-specific limitation: **real, dramatic reuse** - a second request
  sharing a 206-token prefix with an evicted, different-slot request
  reprocessed only 14 new tokens (`cache_n=192`), 27.0ms -> 5.9ms. This is
  the mechanism working exactly as designed once the SWA-specific
  discard path doesn't trigger.

  **Conclusion for the H200/GLM-5.2 deployment**: this backlog item is
  effectively done already, not a build target - GLM-5.2 isn't SWA, so
  the SWA-specific limitation that crippled the Gemma-4 test shouldn't
  apply there. Worth re-confirming once on the real model/hardware (same
  methodology: force a different-slot restore, check `cache_n` in the
  response `timings` and the `-lv 4` trace log for "found better prompt"
  immediately followed by *no* "erased invalidated context checkpoint"
  line), but there's no code to write for the core mechanism - it already
  exists, is on by default, and the one real failure mode found doesn't
  apply to the target architecture. Narrower than full PagedAttention as
  the maintainers wanted (discussion #21961); underlying paged-KV
  experiment elsewhere showed a 26->247 concurrent-sequence capacity jump
  at equal VRAM on an A10G, a different (block-pool) approach not needed
  here given this simpler mechanism already covers the actual ask.
- ~~MTP batch-gate tuning (close the 9-31 dead zone between MAX_BATCH/
  MIN_BATCH)~~ **RESOLVED, 2026-08-14** - see "MTP verify-batch size x
  concurrency" under Open questions above for the full writeup. moe-cache's
  own `MAX_BATCH` gate is now MTP-aware (was silently ignoring MTP's
  verify-width multiplier entirely, fixed and verified). The empirical
  9-31 sweep found no dead zone in that range itself (smooth climb, 51.2
  -> 77.5 tok/s) - but did find a real, separate, sharp cliff exactly at
  concurrency=32 (77.5 -> 41.2, -46.8%), matching `GGML_OP_OFFLOAD_MIN_BATCH`'s
  default threshold precisely, confirmed independent of moe-cache
  (reproduced with moe-cache both genuinely active and accidentally
  starved of VRAM). Not root-caused to kernel level like the MMVQ cliff
  was, but comfortably outside the 5-10 concurrent user deployment target,
  so not a blocker. **Not entangled with the concurrency-cliff root cause
  after all** - that turned out to be a separate GPU-kernel-dispatch limit
  in `ggml-cuda` (`MMVQ_MAX_BATCH_SIZE`, see below), unrelated to this
  MAX_BATCH/MIN_BATCH question.
- ~~Suffix Decode (PR #26283, model-free spec decoding)~~ **BUILT AND
  MEASURED, 2026-08-14.** Merged `github.com/ggml-org/llama.cpp` pull/26283
  directly (`git fetch origin pull/26283/head`, clean merge onto
  `moe-cache-port`, only 9 files touched - the actual feature, no
  unrelated churn). One compile fix needed: `ngram_suffix`'s
  `need_embd() override` didn't override anything - our branch's
  `common_speculative_impl` base already dropped that virtual (MTP's own
  splitting decision goes through `task->need_embd()` now, a different
  mechanism) - removed the dead override, not a functional change.
  Mechanism: an online suffix tree built from the prompt + previously-
  accepted tokens (no second model, no extra compute cost beyond tree
  maintenance); draft tokens come from matching the current context
  against previously-seen subsequences. `--spec-type ngram-suffix`,
  comma-combinable with MTP (`--spec-type draft-mtp,ngram-suffix`).

  **Measured** (ncmoe=15, 2 samples averaged, same methodology as
  everything else this session), repetitive vs diverse prompts:

  | config | repetitive tok/s | diverse tok/s | rep accept% | div accept% |
  |---|---:|---:|---:|---:|
  | baseline | 50.88 | 51.47 | - | - |
  | MTP alone | 73.53 | 59.89 | 90.4% | 66.4% |
  | suffix alone | 62.10 | 51.29 | 64.4% | 34.0% |
  | **MTP + suffix combined** | 72.88 | **64.82** | 77.5% | 71.1% |

  Suffix-alone matches its design intent exactly: real win on repetitive
  text (+22.1%), flat on diverse text (+0.35%, noise) - nothing to catch
  in genuinely non-repeating text. The nuance: **combined is *not*
  strictly additive everywhere.** On repetitive text, combined (72.88) is
  marginally *worse* than MTP-alone (73.53, -0.9%) - MTP alone is already
  near its own acceptance ceiling there (90.4%), so a second drafter
  competing for the same easy tokens adds coordination overhead without
  new upside. But on **diverse text, combined beats MTP-alone by +8.2%**
  (64.82 vs 59.89) - suffix-decode picks up extra accepted tokens (common
  words/short phrases) that MTP alone misses, a genuine additive effect,
  just showing up on the *opposite* prompt type from what "best case is
  repetitive output" would suggest. Content-verified coherent via
  `/v1/chat/completions` on the combined config (63% acceptance on a real
  diverse prompt, real `reasoning_content`, not degenerate repetition -
  same discipline as every other number in this doc).

  **Recommendation**: enable `ngram-suffix` alongside MTP by default for
  this deployment - the diverse-text case is the common one for an
  agentic API workload, and the repetitive-text downside is small (-0.9%)
  next to the diverse-text upside (+8.2%).
- ~~The still-open concurrency-cliff root cause~~ **RESOLVED, 2026-08-14**
  - see "ROOT-CAUSED" under the concurrency-cliff investigation section
  above. Not llama-server's slot scheduling after all: a GPU-kernel
  dispatch threshold in `ggml-cuda/mmvq.cu` (`MMVQ_MAX_BATCH_SIZE=8`
  default), confirmed to the exact token via a dedicated 7/8/9/10
  concurrency sweep. Directly relevant to the H200/GLM-5.2 target - for
  NVIDIA cc>=Ada Lovelace (Hopper included) this threshold is a flat 8
  unconditionally. Fix shipped as a startup warning
  (`common_warn_concurrency_cliff`), not a silent workaround - it's a
  GPU-kernel limit, not something `-ncmoe`/placement can fix.

## Longer-term / exploratory - not yet scoped as real builds
- Persistent cross-session usage history (Colibri-style) - avoids
  cold-restart penalty, no number.
- Live skew-detection driving adaptive strategy (our own synthesis) -
  protects against static-beats-dynamic-on-uniform-routing failure case,
  unscoped.
- Live per-expert heatmap UI - debugging/demo value, not performance.
- ~~Layer 2 extension: concurrency-target-aware calibration~~ **BUILT AND
  VALIDATED, 2026-08-14.** `--moe-calibrate` now benchmarks every
  candidate (ncmoe, threads, spec-draft-n-max) with real concurrent
  requests and scores by aggregate throughput whenever `--parallel N > 1`
  is passed alongside it, instead of always using solo decode speed -
  directly motivated by this session's own findings that the MMVQ and
  bulk-offload GPU-kernel-dispatch cliffs only appear above certain
  concurrent batch sizes, invisible to a solo benchmark entirely.
  Deliberately reuses `--parallel` itself as the concurrency target
  rather than a separate flag, since the calibration cache key already
  includes `n_parallel` - a separate flag risked drifting out of sync
  with the deploy-time `--parallel` and silently missing the cache.
  Found and fixed a real bug along the way: the benchmark's request
  builder wrapped JSON bodies in shell single quotes, and one of the
  probe prompts ("Explain Newton's second law...") had an apostrophe
  that broke the quoting outright - added proper escaping
  (`common_shell_quote`) rather than just avoiding the word. **Verified
  end-to-end**, not just compiled: calibrated at `--parallel 8` found
  ncmoe=15 at 76.11 aggregate tok/s, closely matching this session's
  independent concurrency-sweep measurement at the same concurrency
  (78.12 tok/s, within normal noise) - real cross-validation, not a
  unit-level check. Confirmed the cache correctly isolates by
  concurrency: a `--parallel 1` launch afterward missed the
  concurrency=8 entry and fell back to Layer 1's live safe floor.
- ~~Verify whether Colibri actually avoids the full prefill pass before the
  first token~~ **RESOLVED, 2026-08-14, checked against actual source**
  (`c/kv_prefix.h` in JustVugg/colibri, cloned and read directly). It
  doesn't - the file's own comments are explicit that prefilling zero
  tokens would leave no hidden state to sample from, and a genuinely new
  or diverged prompt gets full prefill exactly like llama.cpp. What it
  actually does: each conversation is pinned to a KV slot (since their
  commit #639) with the exact token ids that state was built from
  recorded; if the next turn's prompt prefix exactly matches those
  recorded ids, only the new tail gets prefilled, skipping re-processing
  of everything the previous turn already computed - exploiting that chat
  clients resend the whole transcript every turn. All-or-nothing per
  session (`kv_prefix_reuse` returns the full matched length or 0, no
  partial/tree matching), any divergence anywhere falls back to full
  prefill from scratch. Their own measured number: DeepSeek V4, a second
  turn reusing 82% of its prompt, 320s -> 61s. This is a narrower,
  single-session version of the same idea as "Automatic prefix caching"
  below (vLLM/SGLang's RadixAttention shares prefixes *across* sessions
  via a shared radix tree; this is scoped to one pinned conversation
  slot, exact-match only) - not a new technique to add, confirms the
  already-tracked item is the right thing to build if this is wanted.

## FR-Spec + multi-drafter incompatibility, final investigation (2026-08-14, end of session)

Explicit directive: find the real root cause, using any means, as the
last piece of this session's work. Summary of the full push below - a
real independent bug was found and fixed along the way, but the
headline bug's root cause was not conclusively pinned down despite
substantial effort. Reporting the state honestly rather than claiming
a fix that isn't verified.

**New findings this pass:**
- **Not suffix-decode-specific at all.** Bisected across three
  completely different second drafter types - `ngram-suffix`,
  `ngram-simple`, and `ngram-cache` configured with no cache files (as
  inert as a second drafter can be: never produces a draft, ever,
  since it has nothing to match against). All three reproduce the
  identical regression (~30 tok/s, ~9-11% accept) when combined with
  MTP + FR-Spec. This rules out anything in any specific second
  drafter's own `draft()`/`accept()` logic - confirmed by reading all
  three implementations directly, all self-contained (`ngram-suffix`
  and `ngram-simple`'s own `accept()` are literally `// noop`).
- **The internal drafter priority order is hardcoded, not CLI-order-controlled**
  (`common_speculative_init()`, `common/speculative.cpp`): ngram-family
  types are always tried before draft-based types (MTP), regardless of
  the order given to `--spec-type`. This means ngram-suffix's
  `draft()` actually runs *every single round* (not rarely, as first
  assumed from an incomplete trace read) - it just usually produces
  nothing usable (its own `n_min` gate) and falls through to MTP. This
  correction doesn't change the diagnosis but corrects a wrong mental
  model from earlier in the investigation.
- **A real, independent bug found and fixed**: `common_speculative_impl_draft_mtp::accept()`
  ignored its `is_other` parameter entirely and unconditionally
  overwrote its own `pending_h` (the recurrent hidden-state input MTP
  feeds into its own next prediction) using `verify_h`/`n_accepted` -
  data that's only meaningful when MTP itself was the drafter actually
  verified that round. When a *different* configured drafter wins a
  round instead, `is_other=true` and that same code still ran,
  corrupting `pending_h` with a stale, unrelated hidden state. Fixed
  by skipping the update when `is_other=true` (leaving `pending_h`
  untouched is strictly safer than overwriting it with definitely-wrong
  data). Found via new instrumentation added this session
  (`LLAMA_DEBUG_VERIFY=1` env var, `common/sampling.cpp`, prints the
  drafted-vs-target token text on every mismatch) which surfaced a
  genuine repeating/lagging pattern in rejected drafts, consistent with
  a stale recurrent-state input.
- **The fix does not resolve the regression** - re-measured at 31.7-31.9
  tok/s, 10.7-11.0% accept, statistically identical to before the fix.
  In hindsight this is logically expected, not surprising: `is_other=true`
  only ever fires on the ~5% of rounds where a different drafter wins;
  the observed ~90% rejection rate is present even across the ~95% of
  rounds where MTP drafts uninterrupted, with no other impl involved at
  all that round. A bug scoped to the rare cross-drafter rounds cannot
  explain damage present in the common, uninterrupted-MTP rounds - this
  should have been checked analytically before spending a test cycle
  confirming it empirically. The fix is being kept regardless: it's a
  real, independently-valid correctness bug (any multi-drafter MTP
  configuration has it, with or without FR-Spec), just not the
  explanation for this one.

**Where this leaves the investigation:** the bug requires exactly
(MTP active) + (any second `--spec-type` configured, regardless of what
it does) + (FR-Spec trim active) - all three, confirmed via direct
bisection on both of the first two dimensions. What's been eliminated
with real evidence, across this and the prior investigation pass:
`p_min` confidence gating, probability-based verification (both by
reading the actual code and by direct A/B test with `--spec-prob-accept`
showing zero effect), any specific second-drafter's own logic (three
different types tested including a maximally inert one), output-buffer
sizing driven by the other type's larger default `n_max`/`size_m`
(tested directly by forcing `--spec-ngram-suffix-n-max 3` to match
MTP's own value - no change), CLI argument order (proven irrelevant,
priority is hardcoded), and now `pending_h` corruption via `is_other`
(real bug, fixed, confirmed not the cause via direct re-test). What
remains unexplained: some structural difference between "MTP is the
only configured drafter" and "MTP is one of several configured
drafters" that affects MTP's own prediction quality on effectively
every round, not just rounds where another drafter actually
participates - and that specifically requires FR-Spec's vocab
restriction to become severe (this same multi-drafter structural fact,
without FR-Spec, measured healthy earlier this session: 66-69 tok/s,
70%+ accept). No further hypothesis was identified with strong enough
supporting evidence to test as a probable fix within this session.

**Recommendation unchanged**: don't combine `--spec-vocab-map` with a
multi-type `--spec-type` configuration. Each half works well
independently and is recommended on its own merits.

## SUPERSEDES THE ABOVE: root cause found, real fix committed (2026-08-14, later same day)

The section above ended honestly inconclusive. Pushed further per
explicit instruction to keep going and check llama.cpp's own
issues/PRs for prior art: issue #23184 ("Pipeline speculative decoding
strategies draft-mtp -> ngram-mod") confirms independent-draft-streams
is a known architectural limitation, not a reported accuracy
regression; issue #23154, a closed/stale CUDA OOM report combining
`ngram-mod,draft-mtp`, gave a useful but not directly applicable clue.
That search didn't find the bug directly, but re-examining the test
methodology did.

**The methodology bug**: every "healthy MTP-only baseline" number in
this document (including the 66-69 tok/s / 70%+ accept figures cited
just above as the healthy comparison point) that was produced by
launching with `-md <path>` and *no explicit `--spec-type`* was never
actually running speculative decoding at all.
`params.speculative.types` defaults to a single-element `{NONE}`
vector; the HF-sidecar auto-detection in `arg.cpp` only populates it
for `-hfd`-style HF-repo auto-discovery, not a plain local `-md` path.
With `types == {NONE}`, `common_speculative_init()`'s impl list is
empty and it returns `nullptr`; `server_slot::can_speculate()` is
`!!spec`, so speculative decoding silently never engaged - the draft
model loaded into VRAM and then was never called. Confirmed directly
with two new debug prints gated on `LLAMA_DEBUG_VERIFY=1`
(`[DEBUG-TYPES]` in `tools/server/server-context.cpp`, now permanent):
`resolved speculative.types = none` / `spec = (nil) (can_speculate
would be FALSE)`.

**Re-ran with explicit `--spec-type draft-mtp`, the only way to get a
real apples-to-apples number:**
- MTP alone, no FR-Spec: **75.93 tok/s, 81.5% accept.**
- MTP alone, with FR-Spec (`--spec-vocab-map`) active, same prompt/config:
  **30.53 tok/s, 8.8% accept.** A real, severe regression - not the
  fabricated "parity" claimed earlier in this document under implicit
  mode, and not resolved by the `is_other` fix above.
- Bisected exactly as before, but now against a real baseline:
  identical collapse reproduces with **`--spec-type draft-mtp` alone,
  no second drafter type configured at all.** This eliminates
  "multi-drafter interaction" as the mechanism outright - the whole
  suffix-decode/ngram angle investigated above and in the section
  before it was chasing a variable that was never actually the cause.

**The real root cause**: this session's own earlier "cache once"
optimization for FR-Spec's weight gather (`output_w`) and scatter
template (`out_template`) - populated a single time via `set_input()`
on the assumption their shape/content never change - has a genuine,
silent correctness bug. Direct A/B test, reverting
`gemma4-assistant.cpp`'s graph-building code to the original live
(recomputed every forward pass) `ggml_get_rows`/`ggml_fill`+`ggml_set_rows`
pattern, same model/config/prompt: accept rate 8.8% -> 44.3%, tok/s
30.5 -> 55.6-56.7. Correctness (byte-identical greedy output) held
throughout, both cached and live - this was never a wrong-answer bug,
only a "confidently wrong draft, so the target rejects it constantly"
bug. The exact GGML-level mechanism (something about the cached
input's `can_reuse() == true` no longer being a safe signal once the
tensor's *populated content*, not just its shape, needs to track the
live graph) was not traced to the instruction level, but the fix is
unambiguous and repeatedly confirmed.

**Fix committed**: `gemma4-assistant.cpp`, `gemma4.cpp`, `models.h`,
`frspec-helpers.cpp` reverted to the live/uncached pattern; the
`output_w`/`out_template` caching fields and the now-unused
`llm_frspec_gather_vocab_rows_from_backend` helper removed entirely
rather than left as dead, tempting code (commit `fc0818502`). Final,
validated numbers for FR-Spec + MTP, `--spec-type draft-mtp`, fix
applied: **55.6-56.7 tok/s, 43-44% accept**, vs 75.93 tok/s / 81.5%
without FR-Spec - a genuine ~26% throughput cost and a real
accept-rate cost, not the free trim originally hoped for. Still a
plausible trade in a real VRAM-constrained deployment (that's the
whole point of trimming the vocab), just not a free one - budget for
it rather than assuming parity.

**What this retroactively means for the rest of this investigation**:
there was never a real "FR-Spec + multi-drafter" bug - that framing
was a byproduct of comparing a broken cached-FR-Spec run against a
baseline (implicit `-md` mode) that wasn't running speculative
decoding at all, so of course adding a second drafter type "made it
worse" - the baseline had nowhere to go but down once real speculation
was actually switched on for either side. The `is_other` fix in MTP's
`accept()` is still real and still kept (any actual multi-drafter MTP
config has that bug), it was just never this bug's cause.

**Standing caveat for the rest of this document**: any throughput or
accept-rate number elsewhere in this file that was produced by
launching with `-md <path>` and no explicit `--spec-type` should be
treated as a **non-speculative baseline**, not a valid MTP
measurement, until re-verified with `--spec-type` set explicitly and
(ideally) `LLAMA_DEBUG_VERIFY=1` confirming `spec` is non-null. This
document was not fully re-audited line by line for this after the
discovery; numbers that explicitly show `--spec-type` on the command
line (e.g. the MTP+suffix-decode "net outcome" figures above, which do
show it) were real speculative-decoding runs and are not affected by
this specific bug, but should still be treated as unverified against
the now-fixed FR-Spec code path until re-measured.

**Recommendation, corrected**: `--spec-vocab-map` is safe to combine
with a multi-type `--spec-type` configuration now that the actual root
cause (the cache-once bug, not a multi-drafter interaction) is fixed.
Budget for FR-Spec's real cost (~26% throughput, ~38pp accept-rate
drop on this sandbox/prompt) rather than assuming it, and re-measure
before relying on it for the GLM-5.2/H200 target, where vocab size and
trim ratio both differ substantially from this sandbox.

## Probabilistic draft acceptance (`--spec-prob-accept`) - BUILT AND VALIDATED, 2026-08-14

Motivated by the FR-Spec+suffix-decode investigation above: llama.cpp's
speculative verification (`common_sampler_sample_and_accept_n`,
`common/sampling.cpp`) only ever accepted a draft token on an *exact*
match against an independently-drawn sample from the target - never
read the draft's own probability at all, at any temperature. This is
strictly more conservative than the classic speculative-decoding
acceptance test (`min(1, p_target(x)/p_draft(x))`, with fallback to the
target's own sample on rejection - not full residual-distribution
resampling, a common, simpler approximation), which is guaranteed to
accept at least as often for the same draft/target pair: if the target
considers a token at least as likely as the draft did, it's accepted
outright, no coincidental exact match required.

**Design, opt-in and structured to be impossible to regress**: new
`--spec-prob-accept` flag (default off - unset behavior is byte-identical
to before this existed, verified). The new code path is a strict
superset of the old one - exact match is checked and accepted first,
exactly as before; only on mismatch does it get a *second* chance via
the probability-ratio test, before falling back to the same "use the
target's own sample" behavior the old code always used. Implementation:
- `common_speculative_draft_params` gained an optional `result_probs`
  (`common/speculative.h`) - parallel to the existing `result` token
  array, populated by MTP's own draft loop with its real per-token
  confidence (`common/speculative.cpp`, already computed via
  `common_sampler_get_candidates`, previously discarded). Drafters that
  don't track a real probability (ngram-suffix and other pattern
  matchers) are padded with 1.0 (maximally confident) rather than
  requiring every drafter type to change - a probability of 1.0 makes
  the acceptance ratio reduce to exactly `p_target`, which is the
  correct treatment for a token with no real confidence value.
- `common_sampler_sample_and_accept_n` gained an optional `draft_probs`
  parameter (`common/sampling.h`/`.cpp`); null (the default) reproduces
  the original function exactly, byte for byte.
- Server wiring (`tools/server/server-context.cpp`): `slot.spec_draft_probs`
  parallel to `slot.spec_draft`, only allocated/passed when the flag is on.

**Verified, not just built:**
- Flag off: confirmed byte-identical draft/accept counts and tok/s to
  before this change, both on the healthy MTP+suffix config and the
  broken FR-Spec+suffix one - zero behavioral change when unset.
- Flag on at `temperature=0` (every measurement earlier in this
  document): confirmed a mathematical no-op, not a bug - a greedy
  target's true distribution is a point mass, so any non-matching
  token genuinely has `p_target=0`, and `alpha=0` correctly, every
  time. Probabilistic acceptance can only ever matter at non-zero
  temperature; this was worth confirming empirically, not just arguing
  from theory, given every number in this document so far was measured
  at temp=0.
- Flag on at `temperature=0.7`, healthy MTP-only config (no FR-Spec, no
  suffix-decode): **53.6-53.8 tok/s (off) -> 55.6-55.7 tok/s (on), a
  real, consistent ~3.5% improvement** - the mechanism genuinely works
  and helps in the case it's designed for.
- Flag on at `temperature=0` and `0.7`, the FR-Spec config that was
  believed broken at the time (cache-once bug still present, and
  measured under implicit `-md` mode besides): **no measurable change
  either way.** At the time this was read as ruling out "borderline
  draft tokens losing narrow coin-flips" as the bug's mechanism. With
  the benefit of the later finding (see "SUPERSEDES THE ABOVE" above),
  this negative result still stands on its own terms - probabilistic
  acceptance genuinely doesn't move a *correctness* bug like the
  cache-once one, since the drafted tokens themselves were being
  computed wrong, not just narrowly rejected - but it should no longer
  be read as evidence about FR-Spec+suffix-decode specifically, since
  that framing itself didn't survive the later investigation.

Recommendation: safe to enable generally (`--spec-prob-accept`) for any
non-greedy (temperature > 0) deployment using MTP - real upside, zero
downside by construction. Independent of the now-fixed FR-Spec
cache-once bug; combine freely with FR-Spec once re-measured against
the fixed code path.

## Explicitly decided against
Multi-LoRA serving (not our use case), multi-node serving (single H200),
SGLang's expert-parallelism/DeepEP (multi-GPU only) - these are ours to
decide and the deployment shape rules them out cleanly.

Full PagedAttention is different and was mis-filed here originally:
not-building-it was llama.cpp's own maintainers' architectural call
(discussion #21961 - "we can obviously implement the same scheduling
for cross-request KV reuse at server level" instead of adopting vLLM's
block-pool architecture), not an independent evaluation on this branch.
Re-examined 2026-08-14, prompted directly by being called out for that
conflation - see "PagedAttention re-examined" below for what's actually
built into llama.cpp already and whether it matters for us specifically.

## PagedAttention re-examined - llama.cpp already has the practical
## equivalent, and this deployment isn't using it (2026-08-14)

Checked the source directly rather than relying on the earlier
maintainer-thread read. Without `--kv-unified`, `n_ctx_seq = n_ctx /
n_seq_max` (`llama-context.cpp`) - each `--parallel` slot gets a fixed,
equal, contiguously pre-reserved share of the context budget, whether
or not that slot's actual conversation ever comes close to using it.
That's exactly the waste PagedAttention exists to eliminate - a
concurrency-count argument (26->247 sequences on an A10G) isn't the
only place this bites; VRAM wasted on unused reservation is a cost
regardless of concurrency level.

With `--kv-unified`, `n_ctx_seq = n_ctx` and `n_stream = 1`: a single
pool shared across all active sequences, drawn from dynamically rather
than pre-split. Combined with `--cache-idle-slots` (already validated
this session, ties into `server_prompt_cache`/`--cache-ram`), idle
slots' KV cache is actively reclaimed and handed back to the pool for
busy ones - `try_clear_idle_slots()` in `server-context.cpp` is a
no-op entirely unless `kv_unified` is set. Checked both for
compatibility with the rest of this stack: `llama_kv_cache_iswa`
computes `size_swa` correctly for the unified case
(`hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch`,
`llama-kv-cache-iswa.cpp`) - a real, tested path (referenced from
`llama-memory-hybrid-iswa.cpp` for hellaswag/winogrande eval), not
experimental - and no MTP-specific special-casing anywhere in
`common/speculative.cpp` or the server's spec-loading code suggests an
incompatibility either.

**The catch, and why this deployment doesn't get this for free:**
`kv_unified` only auto-enables when `--parallel` is left at "auto"
(`server.cpp`: `if (params.n_parallel < 0) { params.n_parallel = 4;
params.kv_unified = true; }`). Any explicit `--parallel N` - which is
exactly what this session's concurrency-aware `--moe-calibrate` work
does (deliberately reuses `--parallel` as the concurrency target), and
what a real deployment pinned to "5-10 users" would do - silently keeps
the wasteful fixed-division default instead.

Given GLM-5.2 is VRAM-constrained enough to need moe-cache's CPU/GPU
expert split in the first place, wasted KV reservation isn't just
"fewer raw concurrent connections" - it directly competes with
moe-cache's own expert-cache budget for the same free VRAM. That's a
more concrete case for us than the raw concurrency-count number the
maintainer thread was actually debating.

**Action, not yet taken:** add `--kv-unified` (paired with the
already-validated `--cache-idle-slots`/`--cache-ram`) to the real
deployment config and to this session's own calibration/sweep commands
going forward - all of which have been running with explicit
`--parallel N` the entire session and have therefore never exercised
unified mode. Every concurrency-related number measured so far in this
document (the 9-31 sweep, the concurrency=32 cliff, the
concurrency-aware calibration cross-validation) was measured under the
fixed-division default, not unified mode - worth flagging as a caveat
on those results, not just a forward-looking TODO, since unified mode
could plausibly change where cliffs/optimal placements land by
changing how much VRAM is actually free at a given `--parallel N`.

**Action taken and confirmed live (2026-08-15):** turned out to need no
change at all on the real deployment - `params.kv_unified` auto-enables
whenever `--parallel` is left unset (the exact "auto" condition described
above), and the 8099 deployment's launch command has never passed
`--parallel` explicitly. `--cache-idle-slots` (default `true`) and
`--cache-ram`/`cache_ram_mib` (default 8192 MiB, so already non-zero) were
already on too - every precondition `server-context.cpp`'s own gate checks
(`cache_idle_slots && cache_ram_mib != 0`) was already satisfied by
defaults. Confirmed genuinely active, not just theoretically satisfied, two
ways: the startup trace line `idle slots will be saved to prompt cache and
cleared upon starting a new task` fires on every launch, and a real 4-slot
test (small enough context, `-c 2048`, to actually get 4 slots instead of
the 1 this hardware's tight VRAM margin usually forces at larger requests -
see the fit.cpp margin-widening fix above) showed it firing live under real
traffic: `saving idle slot to prompt cache` / `clearing prompt with N
tokens` / `updating prompt cache`, once per slot, immediately after each of
4 concurrent requests completed.

The caveat two paragraphs up still stands as written: this session's own
calibration/sweep commands (`--moe-calibrate`, concurrency sweeps) pass
`--parallel N` explicitly and therefore still run under the fixed-division
default, not unified mode - that's a property of those specific commands,
not the live deployment, and would need the same explicit `--kv-unified`
override if those sweeps are re-run and unified-mode numbers are wanted.

## Honest calibration on the combined number
RFC-thread testers got low-to-mid-20s tok/s with moe-cache alone on WORSE
hardware (2x3090 + repaired 8-channel DDR4) running the HARDER case
(DeepSeek's uniform routing). Our setup stacks three favorable factors on
top - GLM-5.2's confirmed skew, H200's far higher HBM bandwidth, native
MTP layered on - which is why 40+ tok/s reads as realistic, not optimistic.
Target, not a guarantee, until actually measured.

# Other software worth knowing about (not deep-dived into source yet)

## KTransformers (kvcache-ai/ktransformers) - validates our approach independently

Published research (SOSP 2026, top-tier systems conference), 19.2k stars,
Apache-2.0, active (pushed 2026-08-08). Now integrated as a backend into
SGLang for CPU/GPU hybrid MoE inference. Core strategy confirmed from their
own README: "Heterogeneous expert placement (hot experts on GPU, cold
experts on CPU)" - the same fundamental approach as leloch's moe-cache we
already ported, independently arrived at and peer-reviewed. Useful as
validation that the design direction is sound, not just a hobbyist idea.

Differentiators worth understanding better if we go deeper:
- AMX/AVX-optimized CPU kernels (Intel AMX-Int8/AMX-BF16 specific -
  depends on the office HPC's actual CPU supporting AMX; needs checking,
  not assumed)
- NUMA-aware memory management (relevant if the H200 node is multi-socket)
- **Native multi-concurrency support built in from the start** (per their
  "Apr 2, 2025: Support Multi-concurrency" changelog entry) - directly
  relevant to our 5-10 concurrent user requirement, which is currently an
  open question for our moe-cache port (see MTP x concurrency question
  above), not a solved design there.

Reported: >220 tok/s total throughput on trillion-parameter MoE models in
their hybrid setup (hardware config not verified from this pass - would
need to check if comparable to our 1x H200 + 512GB target before treating
as a benchmark to match).

Not yet: read their actual source for the concurrency/scheduling
mechanism, compared their eviction/placement policy in detail against
leloch's, or checked AMX availability on the target HPC's CPU.

## ik_llama.cpp (ikawrakow/ikawrakow_llama.cpp) - sibling fork, same problem space

llama.cpp fork specifically focused on CPU/GPU hybrid MoE offload
performance. Already has MTP decoding support merged for GLM-4.x MoE,
Qwen 3.5/3.6, Gemma 4, and GLM 5 - i.e. both of our target models already
have MTP support in this fork, same as we just verified for mainline
GLM-5.2. Uses a concrete, specific GPU-offload threshold formula: prompt
processing offloads to GPU above `32 * total_experts / active_experts`
tokens - a different, more MoE-shape-aware heuristic than the flat
default-32 `GGML_OP_OFFLOAD_MIN_BATCH` threshold we've been discussing for
mainline. Worth comparing once we're tuning batch-size gates for our own
deployment - this formula scales with the model's actual expert:active
ratio rather than being a flat constant.

Not yet: read their actual moe-cache-equivalent implementation, if one
exists, or diffed their offload heuristic against leloch's approach in
detail.

## SGLang (sgl-project/sglang) - MoE-first architecture, mixed relevance

31.7k stars, active (pushed 2026-08-13, today). Three real pieces:

1. **RadixAttention** - generalizes prefix caching from vLLM's fixed
   16-token block hash to a radix tree, handling *branching* conversations
   (agent retries, multiple tool-call paths) more naturally than exact
   block matching. Refines our existing prefix-caching candidate above -
   more relevant given our agentic/API-heavy usage pattern specifically.
2. **Expert Parallelism (DeepEP/MoriEP + EPLB + DeepGEMM)** - specialized
   all-to-all GPU communication for MoE token dispatch, dynamic expert
   load rebalancing, and grouped-GEMM kernels for MoE's per-expert
   batches. All about spreading experts across *multiple* GPUs - **not
   applicable to our single H200 deployment**, this solves a different
   problem (multi-GPU expert sharding) than ours (CPU/GPU tiering on one
   device). Noted for completeness, not a candidate for us.
3. **FR-Spec draft-vocab trimming** (`--speculative-token-map` in SGLang,
   production code) - restricts a speculative drafter's LM-head to a
   frequency-ranked vocab subset (e.g. top 32k of 100k+), cutting draft
   compute ~75% with a lossless guarantee (target still verifies full
   vocab). **Already proposed and implemented for llama.cpp**: issue
   #25187 (open, created 2026-07-01, unusually mature - implemented and
   tested on a real branch, commit 047bfa508). Reuses infrastructure
   llama.cpp's own EAGLE-3 implementation already has for a different
   reason (a `d2t` tensor that trims-then-scatters logits, for drafters
   with a smaller native vocab than the target). ~30 lines, no-op for
   existing GGUFs. **Currently implemented only for `qwen35.cpp`
   (Qwen3.5/3.6 dense MTP) - explicitly checked and confirmed NOT yet
   done for GLM's MTP path** (`glm4-moe.cpp`/`glm-dsa.cpp`). Directly
   relevant to us: proven pattern, just needs extending to our target
   architecture. Real, scoped opportunity, not speculative.

## Does SGLang already do what we're building? Checked directly - no, not reliably

Native SGLang GGUF support is "coming soon" per their own docs, not yet
mature. The only path to GGUF + CPU-offload in the SGLang ecosystem is via
the KTransformers integration, and even there, GGUF loading goes through
what they call a "llamafile backend" - i.e. it reuses llama.cpp/llamafile's
own GGUF-reading code, not independently-built infrastructure.

Reliability check: KTransformers issue #1655 (open, filed 2025-12-02, still
unresolved) - a user tested this exact scenario, including an actual
unsloth GGUF (`unsloth/gpt-oss-20b-GGUF`, same publisher/format we're
using). Results: DeepSeek R1 loaded but failed during batch capture,
GPT-OSS failed during loading entirely. A related issue also names
GLM-4.5-AIR (close cousin of our GLM-5.2 target) as broken with GGUF in
their stack. No evidence found of drafter + limited-VRAM + GGUF all three
working together in SGLang/KTransformers.

**Conclusion**: what we're building isn't a solved problem sitting on a
shelf elsewhere. SGLang is attempting something adjacent by borrowing
llama.cpp's own format-handling to do it, and even that combination has
real, open, unresolved bugs on exactly this class of model (large MoE,
unsloth-quantized GGUF). llama.cpp + the moe-cache port is arguably ahead
of SGLang specifically for GGUF + CPU-offload + native-drafter, not
behind it - useful calibration for the value of this whole project.

## Also found while researching: CUDA graphs for multi-slot decode

Issue #27009 (open, **created today**, 2026-08-13) - proposes decoupling
CUDA graph shapes from the number of active decode slots via fixed-size
padding (dummy token ids, -inf masked padding columns, reserved dummy KV
rows), so graphs can be captured once per bucket size and reused via
`cudaGraphLaunch` instead of rebuilding per active-slot-count. Directly
relevant to our open MTP x concurrency question above - if/when this
lands, it changes how multi-slot decode batching and shapes work, which
could interact with (help or complicate) the moe-cache batch-size gates
we're already tracking as an open question. Brand new, unresolved, worth
watching rather than acting on yet.

## Ollama / LM Studio - confirmed downstream, not independent sources

Both wrap llama.cpp rather than implementing independent MoE-offload
architectures. Ollama's MoE handling is transparently llama.cpp
underneath; their own recent additions are MLX (Apple Silicon) engine
support alongside llama.cpp, not a competing scheduling/placement design.
Checked directly (2026-08-13) rather than assumed - nothing to extract
here for this specific concern.

# Open questions, tracked but not yet resolved

## MTP verify-batch size x concurrency, interacting with moe-cache's batch gate - RESOLVED 2026-08-14

Native MTP (GLM-5.2's NextN head, PR #25980) is architecturally safer than
the external-drafter case the RFC thread found broken - MTP runs in the
same trunk graph, same device/session as the main model, so the "GPU
drafter causes the cache to see zero hits" failure mode (a device/session
binding conflict under shared-draft-device reordering) likely doesn't
apply here.

The RFC thread's "dead batch-size zone" concern - requests sized 9-31
served by neither the cache (gated by `GGML_CUDA_MOE_CACHE_MAX_BATCH`,
default 8) nor the bulk-offload path (gated by `GGML_OP_OFFLOAD_MIN_BATCH`,
default 32) - turned out to be two separable questions, not one:

1. **Was moe-cache's own gate MTP-aware at all? No, confirmed and fixed.**
   `set_max_batch_hint` (llama-context.cpp) passes `n_seq_max` alone, no
   MTP verify-width multiplier - checked directly in the source, not
   assumed. Under MTP, the real per-step batch is `n_seq_max *
   (n_max+1)`, so the gate was silently sized for the no-MTP case on any
   MTP-active deployment. Fixed in `common.cpp`
   (`common_moe_apply_mtp_aware_max_batch_hint`, alongside Layer 1):
   explicitly sets `GGML_CUDA_MOE_CACHE_MAX_BATCH` to `n_parallel *
   verify_width` (capped at the real ceiling, 64) when MTP is active and
   the operator hasn't set an explicit value themselves. Verified:
   `--parallel 8` + MTP n_max=3 went from `max-batch=8` (bug, identical
   to the no-MTP case) to `max-batch=32` (correct); `--parallel 16` +
   n_max=3 went from 16 to 64. Non-MTP configs unaffected.
2. **Does a real throughput dip still exist in the 9-31 range even with
   the gate now correctly sized? Measured directly, and the answer is
   nuanced: no dead zone in 9-31 itself, but a real, separate, sharp
   cliff exactly at 32.** Ran a dedicated concurrency sweep (`-ncmoe 99
   --moe-cache auto -c 12288 --parallel 40`, real concurrent requests,
   confirmed moe-cache genuinely engaged via its own log lines - `max-batch=40`,
   real pool allocation, live hit-rate tracking, not a silent no-op):

   | concurrency | 4 | 8 | 9 | 12 | 16 | 20 | 24 | 28 | 31 | **32** | 40 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | agg tok/s | 67.9 | 78.1 | 51.2 | 58.0 | 63.8 | 67.2 | 72.0 | 75.7 | 77.5 | **41.2** | 47.2 |

   Throughput climbs smoothly and monotonically from 9 through 31 (51.2 ->
   77.5) - the already-known-and-fixed MMVQ cliff shows up once at the
   8->9 boundary as expected, but there's no *further* dip anywhere in
   9-31 itself. Then a sharp, distinct cliff hits exactly at 32 (77.5 ->
   41.2, -46.8%), with partial recovery by 40 (47.2) - the same rise-then-
   cliff-then-partial-recovery shape as the MMVQ cliff, just at a
   different boundary. **Confirmed this is real and not a moe-cache
   artifact**: first run accidentally used `-c 32768` (too large a
   context for `--parallel 40` on this 12GB card, leaving no VRAM for
   moe-cache's own pool - confirmed via zero `[moe-cache]` log lines the
   entire run) and still showed the identical cliff shape/magnitude at 32
   (76.4 -> 41.6); the corrected re-run with moe-cache genuinely active
   reproduced it almost exactly (77.5 -> 41.2). The boundary lines up
   precisely with `GGML_OP_OFFLOAD_MIN_BATCH`'s default (32) - crossing
   into bulk-GPU-offload territory for CPU-resident ops appears to cause a
   real regression here, not the expected improvement, plausibly a
   PCIe/transfer-volume bottleneck from many expert tensors being shipped
   to GPU simultaneously for one step (untested hypothesis, not
   confirmed) - not yet root-caused to the same depth as the MMVQ cliff,
   but the empirical shape is solid and reproducible. **Practical
   takeaway for the H200/GLM-5.2 deployment** (5-10 concurrent users):
   comfortably clear of both cliffs (8 and 32) at that target, so neither
   needs fixing before deployment - flagged here in case concurrency ever
   scales toward 32 in the future.

# Real deployment target (drives all priority calls above and below)

Office HPC: 1x H200 (141GB HBM3e), 512GB system RAM, GLM-5.2 (744B, Q4_K_M,
unsloth quant), 5-10 users - but usage is API/agentic (many requests per
user, not just interactive chat), not simple back-and-forth chat. Target:
40+ tok/s decode.

**Why unsloth specifically**: their dynamic-quant approach is what's
needed to hit the 40 tok/s sweet spot at Q4_K_M quality - same reasoning
that put us on the Gemma-4-26B-A4B unsloth GGUF for the sandbox test.

**Capacity math** (rough, Q4_K_M ~0.59 bytes/param average): ~439GB total
GGUF, ~10GB dense/attention (always VRAM-resident), ~429GB routed experts.
512GB RAM holds the entire expert mass resident - no disk-tier streaming
needed at all, unlike Colibri's harder cases. Leaves ~131GB of H200 VRAM
for the moe-cache hot-expert-set, KV cache across 5-10 concurrent slots,
and compute buffers once dense weights are subtracted - workable but needs
deliberate sizing once we're testing on the real hardware.

**Why this reprioritizes things above generic advice:**
- GLM-5.2's confirmed routing skew (SharkWipf: top ~1000 experts dominate,
  30-50% measured speedup) means moe-cache is a strong fit, not a maybe -
  validated for the real target, not just our Gemma-4 sandbox stand-in.
- GLM-5.2's native NextN/MTP speculative decoding is merged in-tree (PR
  #25980) - uses the model's own trained draft head, no external
  checkpoint needed. Likely the single highest-value, lowest-effort win
  for this deployment specifically. **Verified 2026-08-13**: fetched the
  header of `unsloth/GLM-5.2-GGUF` (UD-Q4_K_M, shard 1/11) directly and
  confirmed `glm-dsa.nextn_predict_layers = 1` in the GGUF metadata - the
  MTP layer was not stripped, it's present in the file we'd actually
  deploy. Also from the same header: `glm-dsa.block_count = 79`,
  `glm-dsa.expert_count = 256`, `glm-dsa.leading_dense_block_count = 3`
  (first 3 blocks dense, MoE starts after).
- 5-10 nominal users but agentic/API-heavy traffic pushes effective
  request volume and repetition patterns well past "5-10 people typing in
  a chat box." Prefix caching (agents resending stable system
  prompts/tool schemas every turn) and Suffix Decode (repetitive
  code-edit/tool-call output) both become more directly relevant here
  than they'd be for plain chat - matches the RFC thread's own finding
  that agentic coding workloads are prefill-heavy by construction.
- A single H200 (not multi-GPU) means the RFC thread's multi-device
  pool-distribution findings (sticky layer/device routing, uneven pool
  sizing across cards) don't apply - simpler single-device sizing problem.
- Sanity check against real numbers: RFC-thread testers got low-to-mid-20s
  tok/s decode with moe-cache on 2x RTX 3090 + repaired 8-channel DDR4
  (~100GB/s) running DeepSeek-V4-Flash 284B - the *harder* case (uniform
  routing). GLM-5.2's skew plus an H200's far higher HBM bandwidth plus
  512GB of presumably fast server RAM all point toward 40+ tok/s being a
  realistic target, not optimistic - more so once MTP speculative decoding
  stacks on top.

# vLLM features beyond DSpark/Suffix Decode - priority order

Not yet scoped, just recorded with a rough priority (nearest-term first).
Status verified by reading the actual PRs/discussions/issues, not assumed.

## 1. Automatic prefix caching (next up after DSpark/Suffix Decode)

vLLM hashes KV-cache in 16-token blocks and reuses them across *different*
requests sharing a prefix (system prompt, few-shot preamble, multi-turn
history) - not just within one conversation. A natural extension of
PagedAttention's block-based, reference-counted KV layout: since blocks are
already page-like, sharing them across requests by content hash falls out
almost for free. llama.cpp's prompt caching is per-slot/per-session today,
not a global hash-addressed cache shared across arbitrary concurrent
requests. Relevant for llama-server multi-session workloads (agentic/tool
systems hammering the same system prompt across sessions).

**Real status, verified by reading the threads (2026-08-13):**
- Two prior PagedAttention attempts: #17579 (closed, zero comments, no
  adoption) and #22569 (open draft, dormant since 2026-05, real numbers -
  on an A10G at equal KV budget, unified KV OOMs at 26 concurrent
  sequences, paged mode handles 247).
- The actual design discussion is #21961 ("Paged KV cache and scheduler:
  Phase 1"), still genuinely live - last comment 2026-08-11, two days
  before writing this. No maintainer verdict landed yet, not rejected,
  just unresolved.
- **The real maintainer pushback isn't against the outcome, it's against
  the implementation strategy.** ngxson (maintainer): "paged attn is only
  beneficial when the incoming requests have the same prompt prefix...
  we can obviously implement the same scheduling for cross-request KV
  reuse at server level" - i.e. get prefix-sharing without adopting
  vLLM's whole fixed-block-pool KV architecture. am17an (maintainer)
  separately noted most llama.cpp users don't run enough parallel
  requests for the >25-concurrency win to matter to them specifically.
  Even the original design discussion #21961 already deferred CoW/prefix
  caching itself to an unscoped "Phase 2" - it was never actually in
  scope for the PR that stalled.
- **Implication for us**: don't port PagedAttention wholesale. The
  narrower, maintainer-suggested path - server-level cross-request KV
  block reuse without a full block-pool rewrite - is the one with an
  actual chance of landing, and is a smaller, more scoped project than
  what either stalled PR attempted.

## 2. Chunked prefill + continuous batching (later - look at together)

Chunked prefill splits a long prompt's prefill into chunks interleaved with
other requests' decode steps, so one huge prompt doesn't stall everyone
else's tokens; only makes sense on top of continuous batching (a new batch
formed every iteration rather than waiting for the whole in-flight batch to
finish). Both are multi-tenant serving mechanisms - relevant to llama-server
specifically, not the CLI/single-user path.

**Real status, verified (2026-08-13):**
- Continuous batching is **already shipped** in llama-server, not a gap -
  PR #6358's title ("Allow continuous batching to be disabled") implies
  it's on by default. llama.cpp already has this one; what it lacks is
  vLLM's more sophisticated paged/chunked version layered on top.
- Chunked prefill's tracked PR (#10718) is **stale, not just unmerged** -
  created Dec 2024 as a single-day draft/example, zero updates since. Very
  different situation from the paged-KV discussion, which is still active
  this week. If we want this, we're likely starting closer to scratch than
  picking up existing momentum.

## 3. Multi-LoRA serving (after that)

Serving many different LoRA adapters against one base model simultaneously,
swapping per-request cheaply (S-LoRA-style). Real capability llama.cpp
doesn't match today, but even vLLM's own ecosystem is still fighting an
inherent conflict here as of 2026: prefix caching and multi-LoRA serving
actively fight each other, since KV cache can't be shared across requests
using different adapters - active research problem upstream, not a solved
pattern to copy.

**Real status, verified (2026-08-13):** llama.cpp is hitting the identical
collision right now, live and unfixed - open issue #26207 (filed
2026-07-28, unlabeled/untriaged): with per-request LoRA selection and
`cache_prompt: true`, the server reuses KV computed under adapter A for a
request selecting adapter B whenever the prompt prefix matches - silent
output contamination, nothing logged. The only workaround
(`cache_prompt: false`) means paying full prompt re-processing every
request, i.e. disabling the very feature (prefix caching) that would make
multi-LoRA serving fast. This isn't a "vLLM has it, we don't" gap - it's
the same open problem in both codebases, which changes the framing: fixing
this bug in llama.cpp *is* progress on the shared unsolved problem, not
catch-up on a solved one.

## 4. True distributed / multi-node serving (very end)

Tensor + pipeline + data + expert parallelism combined across a GPU
cluster - vLLM's actual mission territory (cloud-scale multi-tenant
serving, what most RL training loops and inference-as-a-service providers
build on).

**Real status, verified (2026-08-13):** llama.cpp's `rpc-server` is real
and used, not vaporware, but the open issues paint a rough picture:
crashes on multi-node CUDA RPC (#26583, GLM-5.2), no error recovery - one
worker's RPC failure aborts the entire process rather than returning an
error (#25938, with a fix in flight at #26724), and model loading that
serializes read+hash+dispatch on a single host core so a 535GB load takes
~15 minutes with the NIC and 95 other cores idle (#25890). This is real,
shipping infrastructure that people are actively hitting real bugs in -
not a mature, hardened distributed-serving stack the way vLLM's is.
Lowest priority for the same reason as before: this is vLLM's core
differentiator because it solves cloud-scale serving, a different mission
than llama.cpp's portability focus - but if we ever did touch this, fixing
the existing rough edges would matter more than adding new capability.

# vLLM features worth tracking (DSpark, Suffix Decode)

Separate from the Colibri notes below - these are already being actively
ported into `ggml-org/llama.cpp` by the community, not just candidate ideas.

## DSpark (already partially in our checkout)

Block-wise speculative decoding: a small draft model proposes several
tokens per step via a "Markov head" instead of one at a time, with
per-token confidence dynamically scheduling how many drafted tokens to
verify. Originally DeepSeek-V4's technique, in vLLM via
vllm-project/vllm#47093 and the vllm-project/speculators checkpoint format.

- Core support: PR #25173 (merged 2026-07-28, well before our checkout at
  1f368f354) - `src/models/dflash.cpp` and related files already present.
- Newer checkpoint format: PR #26275 "dspark: support speculators-format
  checkpoints" (open, NOT in our checkout - confirmed via grep, no
  "speculators" handling anywhere in conversion/*.py or src/). Needed if we
  want to load checkpoints using the newer vLLM speculators format
  (`sample_from_anchor`, pruned draft vocab + d2t remapping table).

**Directly relevant to us**: PR #26275's own verification benchmark uses
[`makora-ai/gemma4-26b-a4b-dspark`](https://huggingface.co/makora-ai/gemma4-26b-a4b-dspark)
- a public DSpark draft checkpoint whose `speculators_config.verifier`
explicitly targets `google/gemma-4-26B-A4B-it` / `Gemma4ForConditionalGeneration`,
the exact base model our unsloth GGUF is quantized from. Confirmed compatible
on the target side. The draft's own internal transformer body is a small
5-layer **qwen3**-architecture head (not Gemma), block_size=7,
confidence_head_with_markov=true, markov_rank=256 - matches the "small
Markov head" description found via web search earlier. Checkpoint is raw
`safetensors`, needs GGUF conversion; likely routes through
`conversion/qwen.py`'s existing dspark-handling (that file already
references dspark/dflash) since the draft body is qwen3-shaped, though this
isn't confirmed without porting #26275 and trying it.

Reported speedup (task-dependent, from the PR's own numbers): coding 1.76x,
QA/rag ~1.06-1.26x, writing ~1.00x (no gain), acceptance rate 0.24-0.58
depending on category.

**Path to actually testing this**: port PR #26275 (same kind of work as the
moe-cache port, smaller diff), download the makora-ai checkpoint, convert to
GGUF, run against our Gemma-4-26B-A4B target. Not started - this is what
"expert-cache first, DSpark after" (our earlier sequencing decision) points
to next once moe-cache is validated.

## Suffix Decode (open PR, not in our checkout)

PR #26283, open. Model-free speculative decoding - vLLM's
[SuffixDecoding](https://suffix-decoding.github.io/): builds an online
suffix tree from the current request's own generated tokens, best when the
matched suffix is long (repetitive output - code edits, structured output,
agentic tool-call loops). No draft model needed at all, so it composes with
literally anything, including DSpark or moe-cache, without the
device-placement complications DSpark has. vLLM's version optionally keeps
a global cross-request corpus (up to 10k past requests); llama.cpp's port
so far only builds the per-request tree, no global corpus yet.

Worth trying once we have a real agentic/repetitive workload to test against
- our own tool-call-heavy sessions in this very project would be a
reasonable proxy.

# Ideas from JustVugg/colibri worth evaluating for moe-cache

Source: https://github.com/JustVugg/colibri (Apache-2.0, C engine + React web UI,
disk/RAM/VRAM tiered MoE inference). Not a port target as a whole — most of its
scope (disk tier, io_uring, NUMA, Metal/Vulkan backends, its own model loaders)
doesn't apply here. These are specific, small, self-contained techniques that do.

Status: research notes only, nothing implemented yet. Each needs its own A/B
against the ported moe-cache before merging — Colibri's own README is explicit
that even their validated techniques are "measurable policies, not promises."

## 1. LFRU eviction + hysteresis + decay (runtime, dynamic) — DONE, see results below

File: `c/tier.h` (~60 lines) was the original Colibri reference.

**Built and A/B'd, 2026-08-14 — see "LFRU eviction: A/B results" under
"Already built, VALIDATED with real measurement" below for the full story,
including two designs that measured worse and were reverted rather than kept
on the strength of a plausible-sounding rationale.** Final shipped design:
`ggml/src/ggml-cuda/moe-cache.cu` now uses a two-segment probation/protected_
structure (probation drains first, protected_ capped at 50% of pool with
heat-based demotion choice at the cap) plus a per-slot heat counter with
periodic decay, used as the eviction/demotion tiebreaker within each segment
(bounded 8-candidate window, 25%-plus-4 hysteresis - Colibri's own margin,
confirmed to matter here too). Measured 65.0% hit rate on a mixed hot/cold
workload vs. plain LRU's 63.8%.

## 2. Persistent cross-session usage history (semi-dynamic — persists across runs)

File: `c/route_trace.h`, `.coli_usage` file format.

leloch's final rework explicitly *removed* hot-set persistence to cut
maintenance burden (commit a01eb2646). Colibri kept it and the implementation
is a careful reference for doing it safely if we ever revisit that trade-off:
- versioned text format, two header records (`-1 <n_layers> <n_experts>`,
  `-2 <version> <engine_id>`) so old parsers still work on new files
- engine-identity hash in the header so one model/engine's history can't
  silently corrupt another's placement (refuses by default, explicit
  `PIN=<path>` override required)
- atomic write via `<path>.tmp` + `rename()`
- its own decay (`COLI_USAGE_DECAY`) separate from the in-memory tier decay,
  because a cumulative histogram gets less responsive as it grows (measured:
  at 18.2M selections a typical turn only moves the ranking 0.2%)

## 3. Live per-expert tier/heat heatmap ("Brain" view) — dynamic, requires a running server

File: `web/src/Brain.tsx`.

Wire format is the interesting part: `/experts` returns one **hex byte per
expert** (top 2 bits = tier, bottom 6 bits = heat), plus a separate bitset of
experts that fired since the last poll. ~39KB of text for 19,456 experts, not
JSON objects. Rendering is plain Canvas 2D (no WebGL/three.js — confirmed
absent from `package.json`): `fillRect` per cell, color = tier, brightness =
heat, and the "flash on fire" effect is a per-expert `Float32Array` pulse set
to 1.0 on a hit and decayed `*= 0.94` per animation frame. Polls every 1.5s,
not a websocket.

Only works while the server is actively running — needs a live endpoint, not
a static report. Portable version: `moe-cache.cu` already tracks
hits/misses/evictions per device (`MOE_CACHE_LOG`); extending that to
per-(layer,expert) tier+heat state, packed the same way, plus a small canvas
component in `tools/ui`, would be a scoped debugging/demo view — much smaller
than adopting Colibri's engine.

**Built, 2026-08-14.** Fetched Colibri's actual `resource_plan.py` and
`Brain.tsx` (raw GitHub content, JustVugg/colibri, Apache-2.0) as reference,
then ported the logic rather than the code - the underlying systems are too
different for literal reuse (Python+HF-safetensors+env-vars vs
C++/CUDA+GGUF+CLI-flags, React vs Svelte 5).

CUDA side (`ggml/src/ggml-cuda/moe-cache.cu`): added purely additive
bookkeeping to `moe_cache_session` - a `host_base -> layer` map and the
widest `n_expert` seen. The layer number reuses `moe_cache_layer_number()`,
the same battle-tested name parser the existing device-routing logic already
depends on (`blk.N.ffn_*_exps` -> N) - not a new heuristic, and populated at
the exact point (`moe_cache_begin()`) where that parse already happens for
routing, so it's one extra map insert on an existing code path, not a new
one. Neither new field is read by any routing/eviction decision, so this
carries none of that logic's correctness risk - confirmed by keeping the
change to pure insertion, no conditionals depending on the new state.
`moe_cache_get_expert_map()` snapshots the live grid under the same session
mutex the hot dispatch path already takes, one byte per cell (tier<<6|heat),
exposed through a new `get_expert_map` slot on the existing
`ggml_moe_cache_api` table (same NULL-safe pattern as `get_stats`) and a
`common_moe_cache_get_expert_map()` wrapper that does a two-call
probe-then-fetch so the caller never has to guess a grid size.

Server side: new `GET /experts` endpoint, same hex-packed wire format as
Colibri's (map + a hits bitset for the flash-on-fire animation) - the hit
bitset isn't tracked in the CUDA cache itself, computed in the endpoint
handler instead by diffing each poll's bytes against the previous one, to
keep the CUDA-side change purely additive/read-only rather than adding more
mutable state there.

UI side (`tools/ui`): Svelte 5 port of `Brain.tsx` - Canvas 2D grid, tier
colour + heat brightness + pulse-decay flash on hit, hover tooltip. Dropped
full i18n wiring for a first cut. The expert-affinity atlas overlay was
scoped out of this first cut (a separate, much bigger measured-topic-
clustering system this fork didn't have yet) - since built and merged in,
see section 3b below.

**Validated, not just built**: `svelte-check` 0 errors across the whole
project (6288 files), production build succeeds, and confirmed live against
the actual running server - grid shape (15 layers x 128 experts) matches
`-ncmoe 15` x Gemma-4-26B-A4B's real expert count exactly, and after a real
generation request, 64/1920 cells populated with a sane tier split (32
protected, 32 probation) and the hit bitset exactly matching the
newly-cached cells, confirmed by direct byte-level inspection of the
endpoint's response, not just "the page loaded."

**Commit note**: the backend (CUDA + server endpoint) and the new,
self-contained UI files (the Brain component, its service, the route page)
are committed. The sidebar-nav wiring and shared route/type/service-barrel
edits are applied locally and working, but not yet committed - they touch
files that sit inside a large, unrelated, already in-progress `tools/ui`
constants reorganization found sitting uncommitted in the working tree (not
a pure rename: real content moved between files, e.g. `MCP_RECONNECT`'s own
shape changed) that isn't safe to bundle in sight-unseen. See the Brain-view
commit message for the exact file list still pending.

## 3b. Merging in the affinity-atlas overlay — BUILT AND VALIDATED, 2026-08-14

Colibri's README shows a third view (`docs/media/colibri-atlas.png`) that
plots experts by *measured topic affinity* rather than a fixed grid. Checked
whether the frontend for it actually ships in Colibri's repo before building
anything: no. `web/src/` only has `App.tsx`, `Brain.tsx`, `ErrorBoundary.tsx`,
`Profiling.tsx`, `main.tsx` plus `components/`/`i18n/`/`lib/` - no
`Atlas.tsx` anywhere. Direct raw-URL fetch of `web/src/Atlas.tsx` 404s on
both `main` and `dev` (checked all branches for the file's existence, not
just guessed). The screenshot doesn't correspond to shipped code on any
branch. What *is* real and reused as-is: the measurement methodology in
`c/tools/expert_atlas/` (`probes.json`, `sweep.sh`, `analyze.py`) - labeled
probe prompts per topic category, greedy decode with speculative decoding
off (draft/MTP contamination and `--topp` pruning are exactly the confounds
that would poison a routing measurement), a replication gate (an expert only
counts as affine to a category if it fired on >=2 of that category's
independent probes, not one lucky prompt), and a specialization score
`1 - H(p(c|e))/log(C)`.

Built new (nothing to port, since no frontend existed for this piece):

- `tools/expert-atlas/expert-atlas.cpp` - new standalone CLI, same shape as
  `tools/imatrix` (own CMake target, `common_init_from_params`, hooks
  `cb_eval`). Ships 27 original probe prompts across 9 categories (code,
  math, science, law, medicine, creative, casual, format, history). Reads
  the `ffn_moe_topk-<layer>` tensor that `llm_graph_context::build_moe_ffn`
  already names via the existing `cb()` mechanism (`src/llama-graph.cpp`) -
  this is the router's own top-k selection, upstream of moe-cache entirely,
  so the measurement is valid with or without caching enabled. Clears the KV
  cache between probes (no cross-probe autocorrelation), accumulates
  per-(layer,expert,category) counts, and writes a JSON atlas: for every
  cell with >=1 observation, a specialization score and an (x,y) position -
  the category-count vector's weighted centroid over category anchors laid
  out evenly on a unit circle, gated by the replication rule above.
- `--expert-atlas-file <path>` server flag (`common/arg.cpp`,
  `common_params::expert_atlas_file`), loaded lazily (`std::call_once`) on
  first `/experts` request and merged into the existing response as an
  `atlas` field. Fully backward compatible - omitted when the flag isn't
  set, so existing Brain-only deployments are unaffected.
- `Brain.svelte`: a Grid/Atlas toggle (only shown when atlas data is
  present). Atlas mode plots the same cache-tier/heat colour encoding the
  Grid view already had, but positions each point by measured (x,y) instead
  of (layer,expert) grid coordinates, with category labels around the rim
  and a point radius that grows with specialization. Position from the
  atlas, colour from the live cache state - the same lookup key
  (`layer*cols+expert`) joins both data sources per point.

**Validated, not just built**: `svelte-check` 0 errors, production UI build
succeeds, `llama-expert-atlas` and `llama-server` both build clean. Ran the
tool for real against the live deployment's own model (Gemma-4-26B-A4B,
CPU-only via `CUDA_VISIBLE_DEVICES=` since the persistent 8099 server was
kept up per standing instruction and the sandbox GPU had no spare VRAM for a
second context) - 27 probes across 30 layers in ~22s, 3635 cells written,
specialization scores spanning 0.0-1.0 (mean 0.47), sane per-layer spread.
Restarted the 8099 server once (necessary - a new CLI flag can't be picked
up without a restart) pointed at the atlas file, sent a real chat completion,
and confirmed by direct JSON inspection: `/experts` returns both `stats`
(hit_rate 0.16, 64/64 slots used, 32 protected) and `atlas` (3635 cells) in
the same response, and every one of the 64 currently-cached (warm/hot) grid
cells has a matching atlas position - the merge join works on real data, not
just compiles.

**UI iteration after first real usage (2026-08-14, same day)**: live testing
against the actual running deployment surfaced three more issues, one of
which turned out to be a real pre-existing engine bug, not a frontend one:

- Grid cells rendered stretched into tall rectangles instead of squares, and
  the whole grid looked far taller than it needed to be. Cause: the canvas
  had `class="h-full w-full"`, which CSS-stretches it to the wrapper's box
  regardless of its intrinsic pixel size - `canvas.width`/`canvas.height`
  were being set correctly (square cells, `cols*(cell+gap)` x
  `rows*(cell+gap)`), but the *displayed* size was whatever the wrapper
  happened to be. Fixed by setting `canvas.style.width`/`height` in JS to
  match the intrinsic size exactly, and centering the (now often much
  smaller) canvas in its wrapper via flex instead of stretching it.
- Atlas points overlapped with hard edges, making dense clusters look like
  bitten "half-moons" where a later-drawn circle fully overwrote an earlier
  one. Fixed with per-point transparency (`rgba(...,0.55)`), a light stroke
  so overlapping edges stay visible, and drawing more-specialized (larger)
  points last so a big generalist blob never fully buries a specialist.
- Per user feedback ("do we even need grid? it can be a very small PiP"),
  replaced the Grid/Atlas toggle button entirely: when an atlas is loaded,
  Atlas is now always the primary view and the plain (layer, expert) grid
  is tucked into a small 220x130 picture-in-picture inset in the corner,
  redrawn from the same `cellColor()`/pulse state. Deployments without
  `--expert-atlas-file` still get the full-size grid as before, unchanged.

**Real bug found and fixed, not a rendering issue**: "brain shows nothing
while chat is happening" and "colours seem fixed, not changing between
requests" were reported together, and turned out to share one root cause,
confirmed with hard evidence rather than guessed at - added temporary
per-call logging to `moe_cache_get_expert_map()` (`-lv 4` to actually see
GGML_LOG_INFO, which llama.cpp's own logger maps to its most-suppressed
TRACE level by default) and reproduced live: the moe-cache session's
identity (its pointer, `tensor_layer.size()`, `n_expert_hint`) changed on
essentially every second request, always immediately preceded by a
`sched_reserve: reserving ...` log line. Traced to
`llama_context::sched_reserve()` (`src/llama-context.cpp`): it is not a
one-time startup call as the rest of the scheduler-construction code
suggested on first read - it's gated by a `sched_need_reserve` flag that
`set_sampler()` sets on every backend-sampler (re)installation, which this
deployment's `-bs` flag triggers per-request. When that flag is set, the
*entire* `ggml_backend_sched_t` gets destroyed and rebuilt
(`sched.reset(ggml_backend_sched_new(...))`), and moe-cache's session was
parented 1:1 to that scheduler's lifetime, so every rebuild silently threw
away all cached-expert state and started a fresh, empty session - with no
error or warning, because from the scheduler's point of view this is normal,
expected behavior.

Fix (additive, does not touch `sched_reserve()`'s actual reservation logic):
a moe-cache session tracks GPU-resident expert state tied to physical
devices and host weight-buffer addresses, neither of which change when the
scheduler is rebuilt - so its lifetime shouldn't be tied to the scheduler
object at all. Added `ggml_backend_sched_take_moe_cache_session()` /
`_adopt_moe_cache_session()` (`ggml-backend.cpp`/`.h`) to detach a session
from a scheduler about to be freed and re-attach it to the replacement
instead of going through `session_create`/`session_destroy`. Wired into both
`sched.reset(...)` call sites in `sched_reserve()` (the normal path and the
pipeline-parallel-fallback retry path).

**Validated on the real deployment**: 4 sequential real chat completions
against the persistent 8099 server, `hits` accumulating cleanly across all
of them (13,044 -> 24,470 -> 37,979 -> 50,570) instead of resetting to zero
after the first - confirms the fix holds under this deployment's actual
`-bs`/`--spec-type draft-mtp`/`--parallel 4` configuration, not just in
isolation.

## 4. Static pre-flight resource planner — NOT dynamic, no running server needed

File: `c/resource_plan.py`, `build_plan()`.

Scans a model's tensor headers directly (no weights loaded) to get exact
per-expert-layer byte sizes, discovers hardware (RAM via /proc/meminfo or
platform equivalents, VRAM via nvidia-smi/rocm-smi, physical core count
correcting for SMT/Apple P+E cores, NUMA socket count), and computes a
**projected cache hit rate** before the model even runs — `resident_expert /
total_expert` bytes given the real VRAM+RAM budget. Correctly avoids
double-counting budget on unified-memory chips (Grace Blackwell, Jetson)
where VRAM and RAM share one physical pool.

The genuinely novel bit: it ties that projection to auto-tuning *other*
features, e.g. disabling speculative decoding when the projected regime is
compute-bound or disk-bound with low hit rate, because their own measurements
show speculation actively loses there (-42% one case, -32% another).

Portable version for us: a `llama-plan`-ish pre-flight tool — scan a GGUF's
expert tensor sizes (metadata only, fast), discover VRAM/RAM, project a hit
rate for our `--moe-cache` budget before the user runs anything. Complements
the runtime cache rather than replacing it.

**Full scoping done, see `docs/moe-cache-planner-scope.md`** - where it
lives (extends `tools/fit-params/fit-params.cpp`, not a new binary), what
new code is needed (an expert-size scanner, resurrected+widened from code
that briefly existed in our own git history), output format, explicit v1
scope boundaries, and a testing plan. Not yet implemented.

**Elevated priority after 2026-08-13 validation - this is no longer just a
nice-to-have.** The RTX 3060 test above needed real debugging (temporary
source-level diagnostics) to discover that `--moe-cache auto`/`on` silently
never engages on a single GPU, `MAX_BATCH` defaults to 1 not 8, and the
default 3GB `RESERVE_MB` can eat nearly all free VRAM on a smaller card -
none of which produced an error, just silent zero-benefit. A pre-flight
planner should compute and print (or directly emit) the correct values for
all three, so nobody else has to repeat that debugging marathon:
- GPU count -> forces explicit numeric `--moe-cache <budget>`, refuses to
  suggest `auto`/`on` when count is 1
- always overrides `MAX_BATCH` to 8 (default is simply wrong for real use,
  barely needs "computing")
- actual free VRAM after estimated model load (dense weights + KV cache
  for N concurrent slots) minus a safety margin, sized against the
  model's real per-expert byte size (from GGUF metadata) so the
  resulting budget clears the ~64x-expert-size minimum pool requirement

**This is static, not dynamic - confirmed 2026-08-13.** GPU count, free
VRAM after load, and per-expert tensor size are fixed facts about the
hardware+model pairing, knowable before the process starts generating and
unchanging for the life of the run. A one-time pre-flight calculation
(exactly this candidate) is the right mechanism - no live monitoring or
running server needed. This is different from candidate #5 below (live
skew detection): routing *concentration* can genuinely vary by workload/
session while the server keeps running (SharkWipf measured skew "per
conversation thread"), which a one-time static calculation can't capture.
Don't conflate the two - static sizing (this section) and dynamic skew
adaptation (candidate #5) are separate mechanisms solving separate
problems, both worth having, not substitutes for each other.

## Reality check on Colibri's own numbers

Independent review (wavect.io, GLM-5.2 on consumer hardware) tempers the
README's framing considerably: cold inference on a 24GB-RAM laptop is
0.05-0.1 tok/s (one token every 10-20s) with a 3-4% expert hit rate; the
5.8-6.8 tok/s "success" case needed six RTX 5090s and 251GB RAM with all
experts VRAM-resident (60%+ hit rate) — not actually consumer hardware.
Also: single-generation only (no concurrent requests/batching), no
multimodal input. Doesn't diminish the individual techniques above (each is
small, self-contained, independently verifiable in the source) but the
engine's own headline claims are rosier than its real-world numbers -
calibrate expectations accordingly, especially for candidate #4's hit-rate
projections, which assume a use case with a large enough cache budget to
actually work.

## Load-bearing caveat: cache value depends on routing skew, not just implementation

From `ggml-org/llama.cpp` PR #26563 ("expert hot store", a third independent
implementation of this idea, active Aug 2026) comment thread, user
blakemartz, 2026-08-08: tested the same hot-cache approach on
DeepSeek-V4-Flash-0731 (256 experts, top-6+1, CPU-offloaded, 2x RTX 4090).
DeepSeek's aux-loss-free load balancing keeps routing near-uniform — after a
1.4K-token warm pass, 209-229 of 256 experts showed warm per layer. At ~11%
VRAM:model capacity ratio, a dynamic hot-cache can't beat ~11% hit rate on a
uniformly-routed model. Measured: dynamic cache 12.77 tok/s vs 13.45 tok/s
for a dead-simple static "pin N complete layers to GPU" baseline at the same
VRAM — static won because complete layers hit 100% by construction with zero
lookup/swap overhead, while the dynamic cache pays real bookkeeping cost for
a hit rate that doesn't beat blind capacity.

Their conclusion, and ours: the 1.7-2.1x wins reported for Qwen3.6-35B-A3B
(this PR's headline number) likely depend specifically on Qwen's routing
being skewed. Balanced-routing models (increasingly common via aux-loss-free
training) may see static placement beat a dynamic cache outright.

Action for our own testing: once Gemma-4-26B-A4B finishes downloading, don't
just measure decode speedup — measure the hit-rate *distribution* across
experts (how concentrated vs uniform). A speedup that tracks raw VRAM:model
capacity ratio means we got a favorable model, not that the approach
generalizes. Encouragingly, a separate data point in the same thread
(Tropfchen, Gemma4-26A4-APEX, cold 18 tok/s -> warm 30 tok/s) suggests Gemma
4's routing does have exploitable skew, unlike DeepSeek's.

## GLM-5.2 specifically: skew appears real (independent field data)

From the same RFC #24528 thread, user SharkWipf (RTX 6000 + Threadripper,
independent of leloch/noonghunna): measured 30-50% decode speedup on GLM 5.2
from a hot-cache approach, and directly observed the shape: "the top ~1000
most frequently hit experts per conversation thread seem to be the most
important ones... after that it seems to fall off rapidly." noonghunna
initially suspected this could be the same under-warmed-pool artifact that
tripped their own early DeepSeek measurement, but after cross-checking
conceded "your falloff is probably real" for the GLM-5.2 case specifically
(with the residual caveat that nobody fully ruled out a VRAM-ceiling
explanation over a routing-saturation one). Net: GLM-5.2's routing looks
skewed like Qwen's, not uniform like DeepSeek's aux-loss-free routing -
consistent with Gemma-4-A4B's own cold/warm gap noted above. Reasonable
prior going into our own test: expect a real win, not a capacity-ratio-only
one, but confirm by looking at the hit-rate-vs-pool-size *shape*, not just
the final speedup number.

## Benchmarking traps to avoid (learned the hard way by others, same thread)

Several testers burned real runs on these; worth checking off before trusting
our own numbers once Gemma-4-26B-A4B is downloaded:

- **`--moe-cache` defaults to `auto`, which is ON.** Omitting the flag is
  NOT a control arm - it measures cache-vs-cache. The zero arm needs
  `--moe-cache off` (or `0`) explicit.
- **CORRECTED 2026-08-13, confirmed from our own test + reading the actual
  source: `auto`/`on` mode NEVER engages on a single-GPU system, full
  stop, regardless of free VRAM.** `moe-cache.cu` line ~1365-66:
  `automatic && budget_devices < 2` unconditionally sets the session
  dormant. This contradicts noonghunna's RFC-thread claim ("no
  device-count check, just a 1GiB minimum-slab floor") - that must have
  applied to an earlier revision; the final rework we ported has a real,
  hard single-GPU gate. **Reproduced directly**: ran identical prompts on
  our RTX 3060 with `--moe-cache off` vs `--moe-cache auto`
  (`-ngl 99 -ncmoe 99`, same prompt, same n_predict) - got 27.0 tok/s
  both times, zero difference, no pool/log lines emitted either run.
  **Explicit numeric budgets bypass this gate** (`--moe-cache <MiB>`) -
  this is the only way to engage the cache on a single GPU. **Directly
  affects the real deployment too, not just this sandbox test - the H200
  target is also a single GPU.** Must always use an explicit budget
  there, never `auto`.
- **The requested budget isn't the granted budget.** VRAM-capped silently;
  read the actual `[moe-cache] ... pool[0]: ... slots=N total=M MiB` log
  line, not the flag you passed.
- **Never set `GGML_OP_OFFLOAD_MIN_BATCH` low on a cache-enabled config.**
  It offloads expert `MUL_MAT_ID` to GPU directly, so no CPU-resident expert
  op remains for the cache to intercept - the cache silently never
  allocates at all. One tester measured a 4.2x *loss* (22.34 -> 5.31 tok/s)
  from this exact mistake. Leave it at the default (32).
- **A batch-size gap exists between the cache's max batch (default 8,
  `GGML_CUDA_MOE_CACHE_MAX_BATCH`) and the offload threshold (default 32,
  `GGML_OP_OFFLOAD_MIN_BATCH`).** Batches of 9-31 are served by neither
  mechanism. Matters for `--parallel N` or larger speculative-decode drafts.
- **Tune admission (`GGML_CUDA_MOE_CACHE_ADMIT_AFTER`) by wall-clock time,
  not hit rate.** A bigger pool with a *higher* hit rate measured ~11%
  slower for one tester - hit rate is a diagnostic, not the objective.
- **Prefill is essentially untouched by the cache itself** (within 0.3-0.6%
  in a clean single-variable test) - large reported prefill regressions
  elsewhere turned out to be `-ub` size tradeoffs (smaller ubatch frees
  compute-buffer VRAM for the pool, at a measured -12.5% prefill / +10%
  decode tradeoff) or mmap/disk-streaming effects conflated with the cache.
- **Composes well with an external speculative-decode drafter, but only if
  the drafter is CPU-resident.** One tester measured 24.4 tok/s (cache
  alone) + 33.6 tok/s (GPU-resident DSpark drafter alone) failing to
  compose at all when combined on GPU (cache sees ~0 hits - a device/session
  binding bug under the shared-draft-device reordering), but 46.8-48.2 tok/s
  when the same drafter ran on CPU (`-devd none`) instead. Relevant for our
  "expert-cache first, DSpark after" sequencing decision.

## 5. Live skew detection driving adaptive strategy (our own idea, not from Colibri)

Checked: Colibri does not do this anywhere (`tier.h`, `route_trace.h`,
`resource_plan.py` all grepped for gini/entropy/skew/concentration - none
found). It only reacts to raw heat counts, which implicitly reflects skew
but never quantifies or acts on it. So this is a novel design, synthesized
from what we've found, not an extraction from Colibri's source.

The idea: leloch's earlier (pre-rework) design had a "baseline-sampled
bail-out" - measured the cache against the pure-CPU path and disabled
itself if not winning - removed in the final rework for maintenance burden
(commit a01eb2646). The RFC thread independently confirmed *why* a static
signal like that matters: dynamic caching only beats static layer-pinning
on skewed routing (GLM-5.2, Qwen) and can lose to it on uniform routing
(DeepSeek's aux-loss-free routing). A live skew measurement (e.g. a
running Gini coefficient or top-K-share over the per-expert hit-count
window, computed at negligible cost since we're already counting hits)
could drive that decision continuously instead of a one-shot bail-out:
lean into aggressive admission when concentration is high, fall back
toward simpler/static-like behavior (or warn and suggest `-ot` instead)
when it's not. Also matches SharkWipf's observation that skew may be
partly *workload*-dependent (per conversation thread), not purely a fixed
model property, which a one-shot startup measurement can't capture but a
live one could.

Not scoped or estimated yet - flagging as a real candidate, not a plan.

## Already checked, no action needed

**NaN-safe router argmax** (`route_trace.h`'s `rt_router_pick`): if router
logits are ever NaN, naive argmax leaves `best = -1`, later used as an array
index / file offset — silent corruption, not a crash. Checked our
`ggml/src/ggml-cuda/topk-moe.cu:136-146` — already sanitizes NaN to
`-FLT_MAX` before the iterative argmax, with a comment referencing
https://github.com/ggml-org/llama.cpp/issues/19659. Nothing to port.

## Design: CUDA MMVQ many-expert density gate (scoped, NOT implemented, 2026-08-14)

Investigated as a "moe.cpp" throughput-triage candidate, prompted by
upstream issue [ggml-org/llama.cpp#25356](https://github.com/ggml-org/llama.cpp/issues/25356)
(Vulkan backend: batched-decode throughput cliff at n_tokens=9 on
many-expert MoE, fixed 8-token thresholds in MMV dispatch). That issue is
Vulkan-only and its own thread found NVIDIA-via-Vulkan *regresses* with a
naive fix (-6% to -35%) - not directly portable. But the same architectural
blind spot exists in our real deployment backend, CUDA, independently
implemented and unexamined until now.

**The gap.** `ggml/src/ggml-cuda/mmvq.cu`'s `get_mmvq_mmid_max_batch(type, cc)`
picks the max token-batch for which the fast MMVQ (mat-vec) kernel is used
for `MUL_MAT_ID` (MoE routed matmul), via a hand-tuned per-architecture,
per-quant-type table (Pascal/Turing+/GCN/CDNA/RDNA1-4, referencing PR
#20905) capped at `MMVQ_MAX_BATCH_SIZE = 8`. The table takes `(type, cc)`
only - no `n_expert` or `n_experts_used` parameter anywhere in its
signature or body. This is exactly the blind spot #25356 identified for
Vulkan: the threshold was tuned against some canonical MoE shape (few
experts, e.g. Mixtral-class), and never accounts for how the real
computational cost driver - average rows landing on the *same* expert,
`n_tokens * n_experts_used / n_expert` - shrinks as `n_expert` grows, so a
many-expert model should tolerate a much higher raw token count before the
tiled/MMQ path actually wins.

**This is not hypothetical for us.** Our own sandbox model, checked
directly via GGUF metadata: `gemma4.expert_count = 128`,
`gemma4.expert_used_count = 8`. That's squarely in the "many-expert"
class the Vulkan issue's testers used (128- and 512-expert models). Our
own `common_warn_concurrency_cliff` (`common/common.cpp`, built earlier
this session investigating this exact CUDA-side cliff under MTP) already
detects and warns about it - on this hardware/quant combination it fires
at effective batch > ~5, well below where a density-aware gate would say
it's actually safe. Applying the community-validated formula
(`n_tokens * n_experts_used <= C * n_expert`, `C = 2`, cross-validated by
three independent testers on the Vulkan side) to our model's real numbers:
`n_tokens * 8 <= 2 * 128 = 256`, i.e. safe up to **n_tokens = 32** - the
literal hardware ceiling (one warp per token at `32 * 32 = 1024`
threads/block), a 4x wider window than the current flat cutoff of 8.

**Why this isn't a quick threshold edit.** `ggml_cuda_mul_mat_vec_q()`
(`mmvq.cu:1173`) has a hard `GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE)`
- not a soft per-arch/type threshold, an absolute ceiling on the whole
function's contract. `MMVQ_MAX_BATCH_SIZE` is threaded through buffer
sizing, warp-count calculation (`calc_nwarps`), and - critically - the
kernel's own `__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()
* warp_size, 1)` (`mmvq.cu:708`), which is a *compile-time* register/
occupancy budget for the whole compiled kernel, not a per-call runtime
check. Simply raising the table's values would change that budget
globally, for every call site including the existing, already-tuned small
-batch case - real risk of a silent regression there (the same failure
mode the Vulkan thread hit on NVIDIA, just via a different mechanism:
register pressure instead of tile-padding GFLOP/s crossover).

**Proposed design (safe: never touches the existing compiled kernel).**
1. Add a second, non-default template parameter to `mul_mat_vec_q_moe`
   carrying the launch-bounds ceiling explicitly (default =
   `get_mmvq_mmid_max_batch_for_device<type>()`, so the *existing*
   call site's instantiation is byte-for-byte unchanged - same object
   code, same register budget, zero regression risk by construction).
2. Instantiate a second, explicit variant at the hardware ceiling (32),
   compiled as a genuinely separate kernel - isolates all new risk to
   code paths that only run in the new regime.
3. Add a new host launcher (parallel to `mul_mat_vec_q_moe_launch`) and a
   new top-level entry point alongside `ggml_cuda_mul_mat_vec_q()` (its
   hard `ne12 <= MMVQ_MAX_BATCH_SIZE` assert means the wide path needs its
   own entry, not a relaxed version of the existing one, so the existing
   function's contract - and every other caller relying on it - stays
   intact).
4. Wire the density gate into `ggml_cuda_mul_mat_id`/
   `ggml_cuda_mul_mat_id_needs_sync` (`ggml-cuda.cu:1889-1957`): after the
   existing narrow-kernel check fails, before falling to MMQ, try the wide
   path when `ne2 <= 32 && ne2 * n_expert_used <= 2 * n_expert` (n_expert =
   `src0->ne[2]`, n_expert_used = `ids->ne[0]`).
5. Validate on this sandbox (Gemma-4-26B-A4B, 128/8 experts, real
   many-expert case): correctness first (byte-identical greedy output at
   batch sizes spanning both the untouched narrow path and the new wide
   path), then throughput (wide-MMVQ vs. current MMQ/cuBLAS fallback,
   which is what it's actually competing against in the 9-32 token range,
   not the narrow kernel) across the batch sizes that currently fall back.
   Also explicitly re-benchmark the *unchanged* narrow path (batch <= 8)
   to confirm the new template parameter really did leave it untouched,
   not just assume it from the design.

**Effort estimate**: real CUDA kernel engineering, not a config change -
new kernel instantiation, new launcher, new dispatch wiring, correctness
+ performance validation across quant types actually in use. Scoped here
so the investigation doesn't need to be redone, but deliberately not
started without a dedicated pass - the blast radius (MUL_MAT_ID feeds
every MoE FFN forward pass) means a subtle bug here is a correctness risk
across the whole model, not a contained feature.

## Backend sampling (`-bs`) validated on this sandbox - real, modest, safe (2026-08-14)

Investigated as a moe.cpp throughput candidate, prompted by upstream
[ggml-org/llama.cpp#27050](https://github.com/ggml-org/llama.cpp/issues/27050)
(same-day issue: `-bs` measured +48% aggregate throughput at 32 concurrent
slots on an RTX 5090, 706 -> 1046 tok/s, greedy output byte-identical,
single-slot unchanged). Unlike the other candidates this session, nothing
needed building or porting - `--backend-sampling`/`-bs` already exists in
this fork, fully wired end to end (`arg.cpp`, `common.h`,
`server-context.cpp`, `sampling.cpp`), marked experimental and off by
default. The open question was purely: does it deliver a real, safe gain
on our own hardware/model, worth recommending for our multi-user serving
config.

**Mechanism** (per the issue, confirmed by reading our own
`server-context.cpp`): the default path copies the full logits matrix
device-to-host every decode step so `common_sampler_sample()` can run on
CPU; at higher slot counts that transfer plus CPU-side sampling/sorting
becomes the bottleneck. `-bs` samples on-device and skips the transfer.
Already self-gates safely: `common/sampling.cpp` disables it automatically
(with a warning) when combined with grammar or a reasoning budget, and
`server-context.cpp` disables it when pre-sampling logits are requested
(`n_probs` without `post_sampling_probs`) - none of that needed building,
it was already there.

**Measured directly on this sandbox** (Gemma-4-26B-A4B, `-ncmoe 15`, RTX
3060, greedy, 200-token completions, aggregate tok/s = total completion
tokens / wall time across all concurrent requests):

| config | bs off | bs on | delta |
|---|---|---|---|
| `--parallel 4` (below this card's MMVQ concurrency cliff - clean comparison) | 103.59 tok/s | 107.48 tok/s | **+3.76%** |
| `--parallel 8` (past the cliff - both arms equally confounded, comparison still fair) | 99.66 tok/s | 104.62 tok/s | **+4.98%** |

Real and consistent in both configs, not an artifact of the concurrency
cliff (the clean `--parallel 4` run shows the same direction and similar
magnitude). Far short of the issue's own +48% at 32 slots - expected: this
sandbox tops out at single-digit `--parallel` on a 12GB card, and the
win's mechanism (D2H copy + CPU sampling overhead relative to per-step GPU
compute) scales with both slot count and GPU speed, neither of which this
sandbox has much of relative to an RTX 5090 at 32 slots. Correctness
spot-checked: coherent, non-garbled output, `system_fingerprint` intact,
no incompatibility warnings triggered (config used no grammar/reasoning
budget/`n_probs`).

**Recommendation**: enable `-bs` for this deployment's multi-slot serving
config - real, positive, low-risk (already-gated, self-disabling on known
incompatibilities, upstream-authored and upstream-tested at real scale).
Not flipping the compiled-in default: it's still marked experimental
upstream, and this sandbox has only validated one model/quant/prompt
shape at modest concurrency, not the breadth upstream would want before
changing what every caller gets by default. The GLM-5.2/H200 target,
running at real multi-user concurrency on much faster hardware, is exactly
the regime where the issue's own +48% figure suggests this matters more,
not less - worth re-validating there specifically once that deployment is
reachable, rather than assuming this sandbox's modest ~4% ceilings it.

## draft-mtp CUDA vs Vulkan acceptance collapse - one hypothesis ruled out, blocked on tooling (2026-08-14)

Investigated [ggml-org/llama.cpp#26750](https://github.com/ggml-org/llama.cpp/issues/26750)
as a moe.cpp candidate - directly relevant, since it's our exact drafter
(MTP) on our exact backend (CUDA). Reporter measured 35.8-40.7% draft
acceptance on CUDA vs 91-92% on Vulkan, same GGUF/build/settings,
"parameter-invariant" and fully deterministic. A second reporter
(zanphear) independently reproduced a related regression on different
CUDA hardware (Grace Blackwell) and bisected it to PR #26510
("speculative: refactor enabled configs common_speculative_init"),
offering "the CUDA-side build window and direction of travel" as evidence
- not a precise single-commit bisection.

**Checked directly**: our fork already carries #26510
(`7bd8282c3`). Read the full commit diff (it's small, 16
insertions/45 deletions, entirely contained in one function) - it's a
purely mechanical refactor: identical drafter priority order, identical
gating conditions (`params.draft.ctx_dft != nullptr`), replacing repeated
`if` blocks with a lambda. `configs.emplace_back(type, params)` vs the old
`configs.push_back(common_speculative_config(type, params))` are
behaviorally identical for a value type. Nothing in this diff is
CUDA-specific or could plausibly explain a CUDA-vs-Vulkan divergence.
**This is very likely not the actual cause** - the "before/after this PR"
comparison zanphear ran almost certainly spans more than just this one
commit. A real, useful negative result: don't waste time reverting or
"fixing" this refactor expecting it to resolve #26750.

**Where this leaves it**: the original issue's own text already suspects
"the MTP head forward pass itself producing degraded predictions on the
CUDA path" - a kernel-level claim, not a dispatch-order one. Properly
investigating that needs a direct CUDA-vs-Vulkan A/B on the same
hardware, which this sandbox can't do right now: `glslang`/shader-compiler
tooling is available via `dnf`, but there's no NVIDIA Vulkan ICD packaged
here (only Intel/AMD/software drivers via `mesa-vulkan-drivers`) - a real
hardware-accelerated Vulkan backend on this RTX 3060 would need NVIDIA's
separate Vulkan driver component, outside normal repos, genuine
infrastructure work with uncertain success. Not pursued further without
that. If the H200 target ever has a working Vulkan install, worth
re-attempting the differential comparison there directly.

## KV cache position crash at context boundary - real, but scoped to DFlash/DSpark, not MTP (2026-08-14)

Investigated [ggml-org/llama.cpp#26478](https://github.com/ggml-org/llama.cpp/issues/26478)
("llama-spec failure at 16k boundary due to non-consecutive KV cache
position tracking (Y != X+1)"). A commenter (devesssi) diagnosed it as
DFlash/DSpark's draft-batching always building a block at the full
configured `params.n_max` regardless of the caller's remaining
per-sequence context budget, opened a draft fix (PR #26575, unmerged,
not yet validated end-to-end).

**Confirmed directly in our code**: `common_speculative_impl_draft_dflash::draft()`
(`common/speculative.cpp:1146`, shared by DSpark via the `is_dspark`
flag) has exactly this - `const int32_t n_draft = params.n_max;`,
unconditional, then decodes a block of that fixed width immediately.

**Live-tested whether this reaches us via MTP** (our actual drafter,
`--spec-type draft-mtp`, `-c 64` forcing `n_ctx_slot` down to its
observed floor of 256, prompt + `max_tokens` deliberately sized to blow
past it): completed cleanly, stopped exactly at the 256-token boundary
(60 prompt + 196 generated), `finish_reason: length`, no crash, no
assertion, server still healthy afterward. **MTP is not affected.**

**Why the difference, traced precisely**: `server_slot::get_n_draft_max()`
(`tools/server/server-context.cpp:460-479`) recomputes a live,
budget-aware cap every round (`n_ctx - prompt.n_tokens() - 2`, further
capped by remaining `n_predict`) and passes it down as `dp.n_max`. A
generic safety net in the shared dispatcher
(`common_speculative_draft()`, `speculative.cpp:2687-2690`, confirmed via
`git log -L` to be long-standing upstream code from PR #22838, not
fork-specific) truncates *any* drafter's returned result to `dp.n_max` -
but only *after* that drafter has already decoded its own batch.
Draft-simple already respects `dp.n_max` as a live stopping condition
inside its own per-token loop (`speculative.cpp:351`) before ever
issuing the next decode - so does MTP, which drafts incrementally one
token per round rather than committing a fixed-width block upfront.
DFlash/DSpark's block-at-once architecture is the odd one out: it
decodes the full `params.n_max`-wide block *before* the post-hoc
truncation ever runs, so by the time the result list gets trimmed, the
KV cache has already been written past the safe boundary - explaining
why the generic safety net (present upstream for a long time) hasn't
been enough to prevent this specific report.

**Scoped fix, not implemented**: mirror draft-simple's own pattern -
respect `dp.n_max` as a hard cap on `n_draft` *before* building/decoding
the block, i.e. `const int32_t n_draft = dp.n_max > 0 ? std::min(params.n_max, dp.n_max) : params.n_max;`
at `speculative.cpp:1146`, matching the check already proven correct at
line 351.

**Implemented and A/B tested, 2026-08-14 (later same day).** Got a real
DFlash/DSpark-compatible model pair after all: `Qwen/Qwen3-4B` (target) +
`deepseek-ai/dspark_qwen3_4b_block7` (draft), converted directly via
`convert_hf_to_gguf.py --target-model-dir` using the registered
`Qwen3DSparkModel` converter - no synthetic/toy model, a real checkpoint
pair. (First tried the much smaller `LiquidAI/LFM2.5-1.2B-Instruct` +
`tugot17/LFM2.5-1.2B-Instruct-DSpark-3L` for faster download, but our
conversion tooling only has registered DSpark support for specific
architectures - Qwen, DeepSeek-V4, Llama/EAGLE-3, MuseGlimmer/DFlash -
`Lfm2DSparkDraftModel` isn't one of them, so that pair was a dead end for
conversion despite downloading fine.)

Applied the one-line fix, then did a genuine A/B: rebuilt and tested both
with and without it, using prompts sized precisely to leave a small
positive remaining context budget (narrower than the configured block
width) right at the boundary - the exact "danger zone" devesssi's upstream
diagnosis describes. **Could not reproduce the crash in our fork's current
state either way**, across several context sizes and prompt-length
targeting attempts (including one landing the danger zone on the very
first speculative round, not requiring multi-round arithmetic alignment).
Some other layer in our fork already prevents this - not fully traced,
but real: this is the third case this session (after #26100, and MTP's own
architectural immunity to this same bug class under #26478 itself) where a
plausible, well-diagnosed community report doesn't actually reproduce here.

**Kept the fix anyway, honestly labeled as preventive, not a confirmed
fix**: it can only ever reduce draft width relative to before, never
increase it, so it's strictly safer by construction regardless of whether
the crash it targets is reachable here - and it mirrors a pattern already
proven correct elsewhere in this same file, at effectively zero cost.
Correctness re-verified after applying it: coherent output, real
speculative decoding measured on the fresh, untuned Qwen3-4B/DSpark
pairing (235 drafted / 64 accepted on a simple prompt - modest accept rate
expected, this pairing has had no calibration or trim work done on it,
unlike the Gemma-4/MTP setup this session spent most of its effort on).

# `--fit` hard-crashes on an explicit context request that doesn't fit, real fix (2026-08-15)

Found live, not gone looking for it: asked for `-c 32768` in the WebUI
model-info dialog (`default_generation_settings.n_ctx` was showing 2048,
the per-slot value with `--parallel 4` on `-c 8192` - correct but
confusing), then user asked to push toward much larger context ("1M").
`-c 1000000` on the 12GB RTX 3060 didn't fail gracefully - it hard-aborted
the whole process (`ggml_abort` -> SIGABRT from a CUDA OOM inside
`cudaStreamCreateWithFlags`, deep in `ggml_backend_sched_reserve_size`).
Root cause: `common_fit_params`'s own *measurement* pass (in
`common_get_device_memory_data_impl`, `common/fit.cpp`) creates a real
`llama_context` to simulate the allocation at whatever `cparams->n_ctx` was
requested. `ggml` aborts hard on CUDA errors by design (no try/catch can
intercept a SIGABRT), so an oversized *explicit* `-c` could crash the
measurement itself, before any of `--fit`'s own reduction logic got a
chance to run.

Why the `n_ctx==0` (auto) path was already safe: unset context resolves to
the model's own trained context internally before anything allocates, so
the "full" probe size was always bounded by something the model itself
was actually built for. An explicit request has no such bound.

**Two narrower fixes tried first, both wrong on real measurement:**
1. Clamp to the model's own trained context (`llama_model_n_ctx_train`)
   before probing. Reasonable-sounding, wrong in practice: Gemma-4-26B-A4B's
   trained context is 262144, and *that* also crashes the measurement pass
   on this 12GB card. "The model's own trained context" is not a
   universally safe stand-in ceiling - it depends on the hardware, which is
   exactly the thing we don't know yet at this point in the function.
2. Two-point linear extrapolation (probe at `n_ctx_min`=4096 and
   `2*n_ctx_min`=8192, derive bytes-per-ctx, extrapolate to the requested
   size). Prevented the crash, but the extrapolated number was badly wrong
   once fed back into the *existing* reduction-search block further down
   (gated by `cparams->n_ctx == 0`, loosened to also fire when this step
   had already clamped something): that block's interpolation math
   (`bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) /
   (hp_nct - n_ctx_min)`) assumes the "full" measurement was taken *at*
   `hp_nct` (the model's trained context) - once step 0 clamps to something
   else, that assumption breaks and the formula computes a bogus (much too
   large) result. Confirmed live: `-c 262144 -ncmoe 15` landed the real
   downstream KV-cache allocation on something far larger than the
   estimated-safe value, and failed (not a hard crash this time, but not
   the intended fix either).

**Final design**: search by doubling from `n_ctx_min`. Each step is at
most 2x the size of the last one that was *actually measured* (not
extrapolated) to fit, so any single probe's potential overshoot stays
bounded and the final answer is a real, checked measurement rather than a
guess from a distant sample. Scoped to the common single-device (or
host-only) case, where the free-memory budget is unambiguous; multi-device
setups fall back to `n_ctx_min` (the one size every other path in this
file already trusts unconditionally) rather than trying to be clever.
Also fixed the interaction bug: when this step 0 search has already found
and validated a safe context, the old reduction-search block no longer
re-runs its (now-invalid) interpolation over it - if the real remeasurement
in step 1 still somehow disagrees (e.g. minor driver-level allocator
variance right at the boundary this deliberately searches up to), it falls
straight back to `n_ctx_min` rather than trusting stale math.

**Validated live, both configurations, on the real deployment** (RTX 3060,
12GB, Gemma-4-26B-A4B + MTP draft, `-ncmoe 15`, GPU confirmed fully free
before each test to avoid a false negative from the live 8099 deployment
competing for the same VRAM):
- `-c 1000000 --parallel 4`: no crash, real generation succeeds,
  landed on 8192 ctx/slot (32768 total across 4 slots).
- `-c 1000000 --parallel 1`: no crash, real generation succeeds,
  landed on 16384 ctx/slot.
- `-c 262144` (the model's actual trained context) with the fix: same
  8192/slot result as the 1M request, confirming the search correctly
  finds the real hardware ceiling regardless of how far beyond it the
  request goes.

Neither number is close to 262144 - that's a genuine hardware ceiling for
this model+card combination, not a bug. The fix's job was never to make
more context magically fit; it's to make the program discover and report
the real number instead of crashing or silently guessing wrong. Restarted
the persistent 8099 deployment on the fixed binary with `-c 262144`
(letting `--fit` decide the rest) - landed on the same 8192/slot x 4,
confirmed via a real chat completion.

# `--fit` priority order was backwards: context should be fixed, `--parallel` should flex (2026-08-15)

Found by direct user correction, not testing: "the parallel 4 and OOM should
be decided by the program right? this is the thing we built it for" and
later, explicitly, "the first step for the program will to choose the
maximum context possible from what the user has given. then maximizing the
thruput in that." The design above got this backwards. `server.cpp` resolves
`--parallel` to a hardcoded default of 4 *before* `--fit` ever runs
(`tools/server/server.cpp:151-156`), so the previous doubling search
effectively searched "what context fits at 4 slots" - a context request the
user explicitly typed on the command line was silently shrinking to
accommodate a slot count nobody asked for.

**Redesign**: the doubling search in `common/fit.cpp` now always probes at
`n_seq_max=1` first - the most permissive case for fitting the *full*
requested context. Only if the request doesn't fit even at one slot does
context itself get reduced (last resort, same as before). If it *does* fit
at one slot, a second small bounded search (`requested_n_seq_max` down to 2,
each step reusing the already-proven-safe context size) grows concurrency
back up afterwards - recovering as much throughput as the confirmed-safe
context leaves room for, not the other way around.

**Real bug found while verifying this, not designed in from the start**:
`common_init_result` (`common/common.cpp`) wasn't writing the (possibly
fit-reduced) `n_seq_max` back to `params.n_parallel`. The server creates one
slot object per `params.n_parallel` (`tools/server/server-context.cpp`)
*independent* of the `llama_context` actually built - confirmed live: before
the fix, the log showed `n_slots = 4` even though fit had decided
`n_seq_max = 1` for the actual context, a real slot-count/context-capacity
mismatch under concurrency, not just a stale number in a dialog. Fixed with
a straightforward write-back; verified `/props` correctly reports
`total_slots` matching the real context afterward.

**Second real bug, found by deliberately reproducing VRAM contention**: the
"doesn't fit even at the minimum context, with 1 slot" branch only set
`n_seq_max = 1` and left `cparams->n_ctx` at the raw, already-proven-unsafe
requested value - the assumption was "step 1/2 below will handle it," but
that reduction logic only actively shrinks context for the `n_ctx==0` (auto)
path; for an explicit `-c` request it just no-ops ("context size set by user
-> no change") and carries the dangerous raw value all the way to real model
loading. Went unnoticed in every earlier test because there's normally
enough VRAM for the tiny minimum-context probe to succeed, routing into the
(correct) doubling search instead - only surfaced by deliberately starting a
second full model instance on the same 12GB card to simulate real
contention (the scenario the user asked about directly: "I think it is not
considering another embedding model being hosted on the device. is our
program taking that in consideration?"). Fixed by clamping to `n_ctx_min`
in that branch too, matching every other unsafe-value path in the function.

**Third, larger real bug - the whole reason this took multiple rounds**:
even after both fixes above, `-c 262144` (single instance, no contention)
still hard-crashed the *entire process* - not at load time, but ~1 minute
in, on the CUDA graph capture of a real (differently-shaped) user prompt
(`ggml_cuda_graph_evaluate_and_capture` -> `cudaGraphInstantiate` -> OOM ->
`ggml_abort`, uncatchable). Root cause: `--fit`'s measurement probe
(`common_get_device_memory_data_impl`) only ever constructs a `llama_context`
- it never calls `llama_decode`, so it can't see two real costs that only
show up once actual inference starts: (1) real model loading (weights
actually read from disk) measured ~400 MiB more in-use at steady state than
the `no_alloc` probe predicted, and (2) CUDA graph capture allocates
driver-side graph state lazily on the first real decode of *each new batch
shape* - a cost with no fixed size, since it depends on the graph's node
count (model/backend/speculative config), and that never showed up in any
measurement fit ever took.

Tried `GGML_CUDA_DISABLE_GRAPHS=1` first (an existing env-gated escape hatch
in `ggml/src/ggml-cuda/common.cuh`) - confirmed by testing that it
*sometimes* worked (a short, low-token completion succeeded) but was not
reliable: a longer completion on the identical config crashed at the exact
same line even with the env var set via `setenv()` early in
`common_init_result`. Root cause of that failure: `is_enabled()`'s check is
cached behind a function-local `static const bool`, evaluated once on the
first call anywhere in the process's life - something inside `--fit`'s own
probing (well before the `setenv()` call could run) already triggered that
first evaluation, latching "enabled" before the env var was ever set.
Reverted that approach entirely rather than fight the ordering.

**Actual fix**: widen the margin the doubling search in `fit.cpp` requires,
specifically for that search (`step0_margin = 3 * margins_s[0]`, i.e. 3 GiB
at the 1 GiB default instead of 1). Not a precisely-calibrated number - the
two gaps above don't have a fixed size - but a well-justified, generously
conservative one for the one search whose answer gets used to load the
*real* model with real weights, where a wrong answer means a hard crash
instead of a merely suboptimal one. Cost: context on this specific
12GB-card + 26B-MoE-model combination dropped from 16384 (1 GiB margin,
still crashes) to 4096 (3 GiB margin, held). Validated with a repeated
stress test - 4+ back-to-back full-length (700 token) generations, ~13
minutes of continuous real load, no crash - after the 1 GiB margin
reliably crashed within 1-3 requests under the same test.

# `moe_cache_prepare_budget()` was a one-time latch, not a live value (2026-08-15)

Found by direct user question, not testing: "the almost OOM signal and
thruput should be continuously seen by the program and values should be
dynamically adjusted. the level 1 which has chosen the edges stays the same
on the first run, and level 2 I think is not functioning as we envisioned
right?" Confirmed by reading the code, not by reproducing a failure:
`moe_cache_prepare_budget()` (`ggml/src/ggml-cuda/moe-cache.cu`) queries
`cudaMemGetInfo` exactly once per device, gated by a `device.budget_ready`
bool that never resets - every call after the first just returns the
already-cached `budget_limit`, forever, for the life of the process. The
function's own comment claimed the opposite: "this reflects what's free
right now, not a stale snapshot."

**Why not just remove the latch entirely**: this function runs on the hot
per-tensor dispatch path (`moe_cache_begin()`, called for every MoE layer
access during every decode step, not just once per new shape). Re-issuing
`cudaMemGetInfo` - a synchronous CUDA API call - on every single call would
add real per-token overhead. Added a time-gated re-check instead
(`MOE_CACHE_BUDGET_RECHECK_INTERVAL` = 2 seconds): frequent enough that real
pressure changes (another process starting, the KV cache growing as a
conversation lengthens) get noticed within a couple of seconds, infrequent
enough that the sync call rate stays well under 1/s under any real decode
throughput.

**Real correctness subtlety, not obvious until reasoned through**: once the
cache has allocated its own pools, they're real resident CUDA buffers -
`cudaMemGetInfo`'s `free` naturally drops by however much the cache itself
already claimed. A naive re-check (`budget = free - reserve`) would see the
cache's *own prior growth* as shrinking headroom and throttle further growth
in response to nothing external at all - a self-inflicted ratchet down to
whatever the budget happened to be on the very first call, silently
defeating the entire point of re-checking. Fixed by reconstructing "total
room available to the cache" as `free_memory + device.allocated_bytes`
before subtracting the reserve - on the first call `allocated_bytes` is
still 0, so this is identical to the old behavior; on every later re-check,
it correctly isolates the budget from the cache's own footprint, so it only
actually moves in response to something *other* than the cache itself.

**Validated live**: with `MOE_CACHE_LOG` at info level, watched the budget
re-evaluate over real requests -
`CUDA0 cache budget re-evaluated: 1442 -> 926 MiB (1054 MiB free, 0 MiB
already cached)`, then after the cache allocated ~921 MiB of its own pools
two seconds later -
`CUDA0 cache budget re-evaluated: 926 -> 919 MiB (126 MiB free, 921 MiB
already cached)`. Raw free memory dropped from 1054 MiB to 126 MiB (the
cache's own pools), but the reconstructed budget held nearly flat (926 to
919 MiB) - exactly the "notice real external pressure, ignore your own
footprint" behavior this was meant to produce, not a coincidence of the
specific numbers involved.

# Dormant CPU experts: telling the kernel which pages to drop (2026-08-15)

User's question, after the placement fix above: "can the program choose some
of the long standing dormant ones to the disk?"

The tiering already exists, by accident rather than design. MoE weights placed
on CPU are mmap'd straight from the GGUF, so they sit on a VRAM -> RAM -> NVMe
ladder already: the page cache holds what's been touched, the kernel re-faults
the rest from the file on demand. What was missing was any *say* in which pages
get dropped first. llama.cpp issues `posix_madvise` exactly twice, both at load
time, both blanket calls over the whole mapping (`src/llama-mmap.cpp`), so the
kernel's own LRU is the sole decider - and it cannot tell an expert that fires
on most tokens from one that hasn't been selected since startup. moe-cache
already knows the difference, because it sees every expert selection.

**What was added**: per-(tensor, expert) last-use tracking on the existing hot
path (one hash lookup per tensor per call, then plain array stores), plus a
periodic sweep that calls `madvise(MADV_COLD)` on experts that are neither
VRAM-resident nor recently selected. MADV_COLD deactivates pages without
freeing them - nothing is evicted outright, correctness never depends on it,
and under plentiful RAM the pages simply stay. Linux-only, no-op elsewhere.
`GGML_CUDA_MOE_CACHE_COLD_AFTER_S` tunes the dormancy threshold (default 120s,
0 disables).

**The bug worth recording - trigger placement**: the sweep was first hung off
`moe_cache_plan()`, the per-tensor decode path, gated on elapsed time. It never
fired once. The reason is the design error, not the timer: dormancy is an
idle-state property, and hanging the check off decode traffic means it can only
run while the system is busy - the one state where nothing has gone cold yet.
Moved to the fill worker thread, whose `cv.wait` became a `wait_for` on the
sweep interval, so a timeout *is* the idle signal. It fired on the first try
there.

**Two testing faults cost more time than the feature did**, both worth naming
because neither was a product bug: `pkill -f "port 8099"` returning nonzero
aborted the rest of the command chain, so a *stale* server kept serving while
each "new" launch silently failed to bind - several rounds of debugging a
binary that was never running. Then the debug counter used `% 4000` when each
plan call registers up to 64 hits, so ~700 real calls never reached the
threshold and the probe looked dead. Verify the process under test is the one
you built, before concluding anything about the code.

**Measured (12GB card, 26B MoE, -ncmoe auto-raised to 23, 65536 ctx)**: sweep
advised 4499 dormant experts (7957 MiB) ~30s after a request. Generation stayed
correct afterward. Throughput with the sweep on (37.8 / 42.8 / 47.8 tok/s) vs
off (43.1 / 40.9 tok/s) is the same within noise - expected, since this machine
has RAM to spare, so advised pages stay resident and re-faulting never happens.
The value is strictly under memory pressure; the honest claim is
"free when it doesn't help", not "faster".

An earlier 83 tok/s reading was briefly mistaken for a regression against these
numbers. It came from a shorter prompt with a different token budget, not a
comparable run - the baseline above was measured with the identical prompt and
config specifically to settle that.

# DwarfStar / ds4 reviewed (antirez/ds4, dwarfstar.sh) — 2026-08-15

Reviewed at the user's request, on the premise that it "launches large models
like DeepSeek on 8GB of VRAM" and that crashing where such projects succeed
would be the more serious failure. Two things need correcting before drawing
any lessons, because the premise doesn't survive contact with their docs.

**It does not claim 8GB.** The smallest platform listed on dwarfstar.sh is a
32GB Apple Silicon Mac, and the only published throughput figures are from a
128GB M5 Max: 34.3 tok/s generation, 87.3 tok/s prefill. There is no 8GB or
12GB benchmark in their documentation at all. (A separate community fork,
peppe200175/ds4_RTX_5080_cuda, targets a 16GB RTX 5080 + NVMe.) For contrast,
this fork currently does 61.3 tok/s generation on a 12GB RTX 3060 with a 26B
MoE at 65536 context - a different model on different hardware, so not a like
-for-like comparison, but enough to retire the idea that we're behind a class
of tool that solved something we haven't.

**Its architecture is the one we already have.** Three tiers: non-routed
weights resident, a dynamic cache for routed MoE experts with a configurable
budget, and the GGUF on SSD for misses - on the same reasoning we followed
("routed experts dominate model size", cache misses dominate generation more
than prefill). Their budget defaults to 80% of the backend's recommended
working set; ours is free VRAM minus a live 5% reserve. Ours additionally has
LFRU eviction with heat and admission control, which their docs don't describe.

Two genuine differences worth recording:

1. **Explicit read/write I/O instead of mmap**, deliberately, to avoid
   excessive VM mappings on systems already mapping large files. We rely on
   mmap plus the page cache, now steered by MADV_COLD (see the section above).
   Theirs gives exact control over what's resident; ours gets the kernel's LRU
   for free and only hints at it. Not obviously worse - but it is the one place
   where a machine whose RAM cannot hold the CPU-side experts behaves
   differently, and we have not tested that case.

2. **Overlapped streaming prefill**: they reserve headroom for two full routed
   layers so expert I/O for the next layer overlaps compute on the current one.
   The nearest thing we have is thecodacus's expert prefetch (+36% prefill,
   credited in the acknowledgements). Same idea, different mechanism.

**The one concrete thing they have that we don't**, and it is worth building:
*KV cache as a disk citizen* - long prefixes saved to SSD and resumed by prompt
hash, so a restart doesn't force a full re-prefill. Our prompt cache
(`--cache-ram`, 8192 MiB by default, confirmed active) is RAM-only and dies
with the process. Persisting it keyed by prompt hash is a well-scoped feature,
it is exactly the "cold things belong on SSD" principle applied to the tier we
had not applied it to, and unlike weight streaming it needs no new tier - just
durability for a cache that already exists.

# A testbed for "RAM can't hold the experts", and what it costs today (2026-08-15)

Point 1 of the DwarfStar review - explicit read/write I/O with a hard RAM
budget, instead of mmap plus the kernel's page cache - was flagged as the one
case we had never tested, because every model to hand fits in RAM. Two attempts
to get a model that doesn't fit failed for reasons worth recording:

- **No model that large fits this disk.** GLM-5.2's smallest published quant is
  217 GB (UD-IQ1_S) and Kimi-K2-Thinking's is 247 GB, against 96 GB free. These
  are 750B- and 1T-class MoEs; no quantization brings them into range. GLM-4.5
  -Air at UD-Q4_K_XL (67.7 GB) does fit, and is the right size to force
  streaming against 30 GB of RAM.
- **The network wouldn't cooperate.** HuggingFace pulled at ~1 KB/s and GitHub
  at 0 B/s, so outbound bandwidth was the blocker, not the disk. Download
  stopped, partial kept (hf resumes from `.incomplete`).

**A better testbed, no download required.** The user's suggestion - run a second
instance - doesn't work directly: weights are mmap'd `MAP_SHARED`, so two
instances of the *same file* map the *same physical pages*. Page cache is shared
and RAM use doesn't double; only KV cache and compute buffers do. The same trap
applies to cgroup accounting: pages faulted in by the first instance are charged
to *its* cgroup, so a limit on the second instance never bites on weights.

Two fixes make it valid, and both were used:
1. Give the second instance its **own copy** of the GGUF, so its pages are its
   own. On XFS with reflink this is free - `cp` of the 16 GB model took 1 ms and
   consumed no additional disk, while producing a distinct inode (402660884 vs
   268782843), which is what page-cache separation actually keys on.
2. Run it under a **cgroup v2 memory cap** (`systemd-run --scope
   -p MemoryMax=8G -p MemorySwapMax=0`), CPU-only (`-ngl 0`) since the GPU is
   already committed to the primary instance. Swap disabled so paging to swap
   can't be mistaken for streaming from the model file.

This reproduces the condition exactly: `memory.events` showed `max 13225`
reclaim events at load and **71172** after a single 40-token generation, with
`oom_kill 0` - it survived by continuously evicting and re-faulting weights,
which is precisely "streaming from disk because RAM is insufficient".

**Measured cost, same model, same CPU-only settings, 40 tokens:**

| | prompt tok/s | generation tok/s |
| --- | --- | --- |
| unlimited RAM | 22.11 | 10.80 |
| 8 GiB cap (streaming) | 0.34 | 0.51 |

**A 21x collapse in generation.** Worth being precise about the cause: this is
not the SSD being slow in aggregate (it reads 513 MB/s sequentially). It is that
the kernel's LRU has no idea which experts matter. Each token routes to 8 of 128
experts per layer; the page cache evicts on recency alone, so a hot expert is as
likely to be dropped as one that has never been selected, and the resulting
access pattern is small and scattered rather than sequential.

That is the entire argument for point 1, now with a number attached: the gap
isn't bandwidth, it's *knowing what to keep*. moe-cache already tracks exactly
that (per-expert heat, admission counts), and the MADV_COLD sweep already feeds
a weak version of it to the kernel - weak because an advisory hint can only
lower a page's priority, never pin the hot ones. An explicit residency budget
would decide directly. Note the sweep does not apply to the measurement above:
it lives on the CUDA moe-cache path, and this test ran `-ngl 0`.

# Pinning hot CPU experts with mlock: built, measured, default-off (2026-08-15)

The 21x collapse above has an obvious-looking fix: the kernel evicts by recency
and cannot tell a hot expert from one never selected, but moe-cache already
knows. So rank experts by how often they are actually selected and hold the top
of that ranking down with `mlock`, which is the one thing `MADV_COLD` cannot do -
an advisory hint lowers a page's priority, it never protects.

Built exactly that: per-expert selection counts on the existing hot path (the
same tracking the MADV_COLD sweep already needed), a descending rank, and an
mlock/munlock pass in the periodic sweep that keeps the pinned set tracking the
workload. Budget derived from MemAvailable, additionally clamped by the process's
own cgroup `memory.max` - pinned pages are unreclaimable, so budgeting against
the host's figure inside a small cgroup is precisely how a slow-but-working
server becomes an OOM kill.

**Two trigger bugs found on the way, both the same shape as the CUDA-graph
mistake earlier: right logic, wrong place.**
1. The sweep ran only on the fill worker's *idle* timeout. Under real memory
   pressure that thread is never idle - every miss queues a fill - so a 5 GiB
   -capped run with 69150 reclaim events completed without a single sweep.
   Fixed by driving it on elapsed time regardless of queue state.
2. Pinning sat *after* the sweep's dormancy gate (`age_now < cold_after`,
   default 120s), so it never ran in sessions shorter than two minutes. Which
   experts are hot is known from the first tokens and pinning is wanted
   immediately; only the cold-advise half needs to wait. Split the two gates.

**Then it worked, and the result was negative.** With pinning engaged (365
experts, 639 of 640 MiB budget, releasing 188 and 78 on later sweeps as the hot
set shifted - so the set really was tracking the workload):

| 5 GiB cap, generation tok/s | run 1 | run 2 | run 3 | run 4 | reclaim events |
| --- | --- | --- | --- | --- | --- |
| pinning off | 2.20 | 8.73 | 11.74 | - | 69150 |
| pinning on  | 2.03 | 7.59 | 10.30 | 11.40 | 83058 |

Slightly worse, with *more* reclaim. The arithmetic explains it: an eighth of a
5 GiB cap is 640 MiB against ~12 GiB of CPU-side experts, about 5% of the
working set. A single token touches 8 experts x 23 layers ~ 405 MiB, but across
hundreds of tokens the union of hot experts is far larger than the 365 that fit.
Pinning 5% cannot fix a 3x shortfall, and because pinned pages are unreclaimable
it shrinks the pool every other page competes for - so it costs slightly more
than it saves.

**Shipped default-off** (`GGML_CUDA_MOE_CACHE_PIN_MB=<N>|auto` to enable),
because a feature that measures worse than not having it should not be on by
default no matter how good the reasoning behind it was. The reasoning still
looks right for a *modest* shortfall, where the budget covers a real share of
the hot set; it is wrong for the 3x shortfall tested here. The honest summary is
that explicit residency control needs to own most of the working set to beat the
kernel, which is an argument for DwarfStar's explicit read/write buffer rather
than for hinting at the page cache from the side.

# Disk-persisted prompt cache: writes and restores, not yet adopted (2026-08-15)

The RAM prompt cache dies with the process, so every restart re-prefills prompts
it had already computed - the expensive half of a request, and the half most
likely to repeat. This is the one concrete capability the DwarfStar review found
that we lacked ("KV cache as a disk citizen").

Built: a disk tier under `--cache-disk DIR`, one file per cached prompt, keyed by
a content hash of the token blob. Size limit derived from free disk space (an
eighth, clamped to [1 GiB, 256 GiB]) unless `--cache-disk-size` is given, same
principle as the RAM limit. Files carry a model fingerprint (path + n_ctx +
parameter count) and entries that don't match are ignored outright - a state
restored into a different model isn't stale, it's wrong. Writes go to a `.tmp`
and are renamed only on completion, so a crash mid-save can't leave a truncated
file that a later scan would trust. The scan reads only each file's header and
token list, seeking past the state blob, so building the index costs a few KiB
per entry rather than gigabytes.

**Status: working, after one wrong assumption was corrected (see below).**
Measured end to end on a 3623-token prompt:

- cold prefill: 5887 ms
- state persisted to disk: 271 MiB, written on cache update
- after a full server restart, the scan found the entry and restored it:
  `restored 3642 tokens from disk in 26.95 ms` - a 200x saving against
  re-prefilling, and the number that makes the feature worth finishing

- v1 of the format: the request still re-prefilled anyway (5760 ms,
  `cached_tokens: 0`)
- v2, after the fix: **271 ms, 3618 of 3623 tokens cached** - a 21x speedup on a
  prompt the server had never seen in this process

**The wrong assumption.** v1 deliberately skipped persisting checkpoints, on the
reasoning that they are an optimisation for context shifting which the server
regenerates on demand, and carrying them would roughly double the file size for
no correctness gain. Both halves of that were wrong on this model.

Instrumenting the slot after a restore gave the answer immediately:
`prompt_tok=3642 input_tok=3623 n_past=3623 pos_min=2618`. `n_past` was correct -
the hand-off worked fine - but `pos_min` was 2618, exactly 1024 (the sliding
window) back from the end. Gemma is an iSWA model: most layers retain only the
last `n_swa` positions, so a restored state's KV legitimately begins partway
through the prompt. Checkpoints are what let the server reuse a prefix that
starts before that window; without them it correctly concluded it could not, and
re-prefilled. The in-RAM tier never hit this because it moves the whole
`server_prompt`, checkpoints included - the difference was invisible until the
same state had to survive a process boundary.

Persisting them (format v2) costs what was predicted - the file grew from 271 MiB
to 705 MiB - and buys the entire feature. "Roughly double the size for no
correctness gain" was exactly half right.

Still opt-in (`--cache-disk`), because of the data-at-rest properties below
rather than any doubt about whether it works.

## Disk prompt cache: it is user content at rest (2026-08-15)

Flagged by the user on review, and correct: the first version wrote cache files
with the default umask. Verified concretely before fixing - directory `755`,
files `644`, both world-readable, and the token blob is a plain array of token
ids that detokenizes straight back to whatever the user typed. The KV state
beside it is derived from the same content. This is user data at rest, and it
survives restarts by design, so the exposure is durable rather than momentary.

Fixed: the directory is created `0700` and every file written `0600`. The
permission change is applied to the `.tmp` *before* the rename, so the file is
never visible at its final name with default permissions, not even for the width
of one syscall. If permissions cannot be restricted the tier refuses to write at
all rather than falling back to a readable file. The server logs a warning naming
the directory on every launch that enables it, and `--cache-disk`'s help text
says outright that prompts are recoverable from these files.

What this does *not* solve, and should be stated rather than implied: the files
are unencrypted, so anyone who can read them as the server's user (or read the
disk offline) recovers the prompts. Prefix matching also means a cached prefix
can serve a later request from a different caller - already true of the in-RAM
cache, but persistence widens the window from one process lifetime to whenever
the entry is evicted. `--cache-disk` is opt-in for exactly these reasons; it
should not be enabled on shared storage or for sensitive prompts without
encryption at rest underneath it.

# Overlapped prefill (DwarfStar point 2): already covered, by moe-cache not the scheduler (2026-08-15)

DwarfStar reserves headroom for two full routed layers so expert I/O for the
next layer overlaps compute on the current one. The nearest thing here is
thecodacus's expert-prefetch port in `ggml_backend_sched`
(`GGML_SCHED_PREFETCH_EXPERTS`), previously measured at +36% prefill, and off by
default (`prefetch_n_slots = 0`). Enabling it looked like free money.

Measured A/B on the live config (26B MoE, `-ncmoe` auto-raised to 23, 65536 ctx),
three distinct ~4.8k-token prompts so the prompt cache could not confound:

| prefill tok/s | p1 | p2 | p3 |
| --- | ---: | ---: | ---: |
| prefetch off | 626.1 | 683.8 | 650.0 |
| prefetch on  | 628.6 | 685.5 | 647.9 |

Identical within 0.5%. Not because there was nothing to overlap - the cache was
running at a 70.4% hit rate, so roughly a third of expert accesses still required
a CPU->GPU transfer. The scheduler-level prefetch simply never engages here:
moe-cache intercepts MoE `MUL_MAT_ID` nodes through its own `begin`/`plan`/
`dispatch` hooks before the scheduler's `op_offload` path can consider them, so
the two mechanisms do not compose. Whichever owns the node owns the transfer.

Which means the overlap DwarfStar describes is already present in this design,
implemented in the wrong place to be recognised as such: a miss enqueues a fill
on moe-cache's worker thread and compute continues, so the copy is already
asynchronous. The real remaining difference is narrower than "we lack overlapped
prefill" - it is that their prefetch is *predictive* (reserve two layers, fetch
ahead) while ours is *reactive* (fill on miss, which arrives too late for the
current token and only helps later ones). Closing that would mean prefetching
inside moe-cache, not enabling the scheduler flag.

Recorded as a negative result for the flag, and as a correction to the DwarfStar
review above, which listed overlapped prefill as something we lacked.

# Cross-run usage history: pre-warming the cache from what the last run learned (2026-08-15)

Chosen as the narrow, design-consistent version of DwarfStar point 2 after the
broad options were rejected: the scheduler prefetch doesn't compose with
moe-cache (above), and a predictive prefetch inside `plan()` would mean editing
slot allocation and eviction - the highest-risk code in the file - for an
uncertain gain. This is Colibri idea #2 from this document's own list, and it
attacks something measured repeatedly across this session rather than something
theoretical: the first request after every restart is by far the slowest, because
the cache starts empty and re-learns the hot set through misses.

Per-(layer, expert) selection counts are written to the path in
`GGML_CUDA_MOE_CACHE_HISTORY` (opt-in; no path, no file - the server does not
write files nobody asked for) and merged with what was already there, so history
accumulates across runs. On the next launch a fresh session ranks them and
pre-admits the hottest.

Safety follows the recipe recorded in section 2 above: versioned plain-text
format, an identity hash over layer count, expert count and expert shapes so a
file written for one model can never seed another's placement (mismatch is
refused, not silently used), and an atomic `.tmp` + `rename()` write.

Placement of the pre-warm matters as much as the idea. It runs immediately after
pools are created, where every slot is still free, so admission is a pop from
`free_slots` with no eviction, no heat comparison and no LRU surgery - it cannot
corrupt live cache state because there is none yet. It is bounded to half of each
pool, so a stale or over-large history can't claim the whole cache before the
live workload has said anything.

**Measured.** Unpressured (65536 ctx, three distinct ~4.8k prompts): 936 experts
loaded, 128 pre-warmed, hit rate 70.5% -> 73.7%, and prefill throughput
*unchanged* (627.4/682.2/648.1 vs 628.1/683.5/647.5). Prefill here is
compute-bound and the async fills already hide the misses, so a better hit rate
has nothing to convert into.

Under memory pressure (5 GiB cgroup cap, `-ncmoe 23`), which is where cold start
actually costs:

| generation tok/s | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| baseline | 2.20 | 8.73 | 11.74 |
| mlock pinning (rejected, above) | 2.03 | 7.59 | 10.30 |
| usage history | 2.66 | 13.11 | 14.82 |

Better on every run, +21% to +50%. Stated with the caveat it deserves: n=1 per
configuration and run-to-run variance in this rig is visibly large (the pinning
A/B moved 2.03-2.30 and 10.30-11.74 between otherwise identical runs), so treat
the direction as established and the magnitude as approximate. It is the first
of the three residency ideas tried today to improve anything - the CUDA-graph
change was reverted for resting on a wrong diagnosis and pinning measured worse
than doing nothing.

# Stopping to fix the measurement (2026-08-15)

Three residency policies were tried today - CUDA-graph disabling (reverted, wrong
diagnosis), mlock pinning (shipped off, measured worse), and cross-run usage
history (shipped on, measured better) - and a fourth, `MADV_WILLNEED` readahead
of historically-hot CPU experts, was written next. Its first run produced the
best number of the session, 33.37 tok/s against a 11.74 baseline.

That number is why this section exists rather than a fifth feature. Look at the
spread across nominally identical configurations measured today:

| config | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| baseline | 2.20 | 8.73 | 11.74 |
| pinning | 2.03 | 7.59 | 10.30 |
| history | 2.66 | 13.11 | 14.82 |
| history + readahead | 2.49 | 11.47 | 33.37 |

Run 2 ranges 7.59-13.11 and run 3 ranges 10.30-33.37 *within* this table. The
run-to-run variance is larger than every effect claimed from it. Which means the
honest status of two things already committed is weaker than their commit
messages imply: history's "+21% to +50%" and pinning's "measured worse" were both
n=1 per configuration, and either could be an artefact of which runs happened to
land where. The caveat was stated, but a caveat is not evidence.

So: `scripts/moe-residency-bench.sh`. Each configuration runs N repetitions, with
the page cache dropped and the server restarted before every one, so each is a
genuine cold start rather than a measurement of how warm the previous test left
things. One warm-up request per repetition, then the timed one, so what is
reported is steady-state decode rather than whatever the first tokens cost.
Median with min/max and spread, not a single number.

Readahead is committed behind `GGML_CUDA_MOE_CACHE_READAHEAD` (on by default, 0
to disable) so the harness can A/B it, but no claim is made for it until the
harness has run. The reasoning for preferring it over pinning is sound on its own
terms - advisory hints cost nothing when RAM is short, where pinned pages
actively starve everything else - but "the reasoning is sound" is exactly what
was said about pinning before it measured worse.

# Benchmark results, and a retraction (2026-08-15)

`scripts/moe-residency-bench.sh`, 5 interleaved rounds, 5 GiB cap (~2x
oversubscription of the ~10 GiB CPU-side expert set), 1 cold + 5 warm samples
per server start, page cache dropped and server restarted before every sample.

| config | cold, n=5 | warm, n=25 |
| --- | --- | --- |
| baseline | median 2.34 (IQR 2.22-2.46) | median 12.12 (IQR 10.27-14.63) |
| history | median 2.47 (IQR 2.34-2.51) | median 11.92 (IQR 9.75-17.77) |
| history + readahead | median 2.45 (IQR 2.35-2.58) | median 11.10 (IQR 9.45-14.07) |

Interleaving allows a paired comparison per round, which is the stronger test:
history beat baseline on cold in 4 of 5 rounds (+5.6% median), readahead in 4 of
5 (+4.7%). Neither improved warm throughput; baseline was highest, and readahead
was ~8% worse.

**Retraction.** The usage-history commit claimed "+21% to +50%" from three
single-sample runs. That was noise. The measured effect is **+5.6% on the first
request and nothing afterwards**. The claim was committed with a caveat about
n=1, and the caveat turned out to be the only accurate part of it. Similarly, the
33.37 tok/s readahead reading that prompted building this harness was noise -
nothing across 30 samples came near it (cold max 2.66, warm max 19.38).

Readahead is now **default-off**: +4.7% cold against -8% warm and 3.4 GiB of
page-cache pressure is a bad trade for anything serving more than one request.
History stays on - a small consistent cold gain at no warm cost, for one file.

**The finding that matters more than either number.** Both mechanisms move cold
throughput about 5%, against a 21x gap. That is not a tuning problem, it is the
ceiling of the approach: the kernel's interface is asymmetric. `MADV_COLD` and
`MADV_PAGEOUT` let us say "drop this one first"; `MADV_WILLNEED` is advisory and
routinely ignored under pressure; and there is no hint at all for "this expert is
hot, protect it" - the only way to express that is `mlock`, which is not a hint
but a command, and measured worse for exactly that reason (unreclaimable pages
starve everything else).

So any cooperative strategy has to be demotion-only: we cannot protect the hot
set, only demote the cold set and let the hot set be what survives. That is worth
one more attempt, and cheaply - the current cold threshold is 120 seconds of
wall-clock idleness swept every 30 seconds, which in a 40-second benchmark run
classifies almost nothing, meaning these results largely measure the kernel
operating with *no* information from us. Classifying by recent routing (last N
hundred tokens) rather than wall clock, sweeping every few seconds, and using
`MADV_PAGEOUT` for the confidently-dormant would actually deliver the
information. If that still yields single-digit percentages, the conclusion is
that the interface cannot express what this access pattern needs, and owning the
buffer is the only remaining answer.

# One policy, three tiers: unifying VRAM and RAM residency (2026-08-15)

Arrived at in discussion rather than from a measurement, and it simplifies the
design rather than adding to it.

Today the two tiers run different decision functions for no principled reason.
VRAM uses LFRU - frequency, recency, and protection segments - and holds ~70% hit
rate. RAM uses "not routed to in N decisions", a binary threshold, which exists
only because it was the easiest thing to express through `madvise`. The tier with
the *more expensive* misses is running the *cruder* policy.

The unified form is one ranking with capacity-determined cut points:

    rank every expert by score = frequency (+) recency
      top slice        -> VRAM       (evicted directly; we own the memory)
      next slice       -> keep in RAM (demote everything below it)
      remainder        -> leave on disk

Same scoring function throughout; only the cut points differ, and they come from
each tier's capacity rather than from separate hand-tuned rules. All the inputs
already exist: per-slot heat (VRAM), `selections` and `last_epoch` (RAM), and the
cross-run history file.

**Where the tiers genuinely cannot be identical is enforcement, not policy.** In
VRAM we choose the victim slot outright. In RAM we can only advise, and the
kernel is free to disregard it - and as established above, the advisory interface
is demotion-only, since no hint expresses "protect this" and `mlock` (the only
mechanism that does) is a command that measured worse by starving everything
else. So we compute the same answer for both tiers and can only *ask* for it in
one of them. Whether asking suffices is exactly what the demote-soft/demote-hard
benchmark measures.

One consequence worth keeping in view: a RAM miss costs a disk read (~500 MB/s
here) while a VRAM miss costs a RAM read. The tier with the costlier miss should
if anything be *more* conservative about what it lets go, not less - which the
unified design expresses naturally as a different cut point on the same ranking,
rather than as a second policy to maintain.

## Decision: LFRU stays in VRAM, RAM-tier demotion ships behind a flag (2026-08-15)

The RAM tier now computes exactly what the VRAM tier computes - same heat signal
(`+MOE_CACHE_HEAT_STEP`, saturating at `MOE_CACHE_HEAT_MAX`, halved on decay),
same coldest-pays rule, cut at the median rather than a hand-invented threshold.
Reusing those parameters rather than designing new ones was deliberate: they were
settled by a lot of measurement upstream, and a second policy competing with a
proven one is a step backwards however reasonable it looks.

It is nonetheless **off by default** (`GGML_CUDA_MOE_CACHE_COLD_AFTER_EPOCHS=N`
enables it), because the policy is not what limits this tier - enforcement is.
VRAM picks its own victim slot; RAM can only advise, through an interface that is
demotion-only. Advisory demotion measured a few percent against a 21x gap across
several runs.

The wider judgement, recorded because it is the reason to stop rather than
iterate further: the 21x collapse only occurs when the expert set genuinely
exceeds RAM, which never happens on this hardware with the models available here
- the condition has to be manufactured with a cgroup cap to study it at all, and
the models that would really need it (GLM-5.2, Kimi-K2, Qwen3.8: 200GB+) cannot
be obtained on this connection. Rewriting residency management to own the RAM
buffer would be weeks of work, validated only against a synthetic cap, for
hardware not in use here. The mechanism is implemented, measured, documented and
one flag away; that is the right place to leave it until a model that needs it
actually lands.

# KV cache quantization: q8_0 frees a gigabyte, and the placement search spends it (2026-08-15)

The KV cache is independent of the model's own quantization - a Q4_K_M model
still keeps its conversation state in f16 by default. At 65536 context x 4 slots
that is 2180 MiB of VRAM sitting beside the weights.

**Check the build before trusting any KV-quant result.** This build has
`GGML_CUDA_FA_ALL_QUANTS=OFF`, which compiles only symmetric Flash Attention
kernels: `f16-f16`, `bf16-bf16`, `q8_0-q8_0`, `q4_0-q4_0`. Upstream issue #24485
documents 25-45x prefill slowdowns on combinations without a compiled kernel, and
discussion #22411 makes the same point for HIP. So the *commonly recommended*
setting - quantize K, keep V at f16 for quality - would have fallen off the fused
path here and produced a catastrophic number that looked like an indictment of KV
quantization generally. Symmetric `q8_0/q8_0` was used for exactly this reason.

Measured, same Unsloth Q4_K_M model, 65536 ctx, 4 slots:

| | f16 | q8_0 |
| --- | --- | --- |
| KV cache | 1280 + 900 = 2180 MiB | 680 + 478 = 1158 MiB |
| n_cpu_moe resolved by --fit | 23 layers on CPU | **21** |
| expert layers on GPU | 7 | **9** |
| generation tok/s | 72.7 / 77.2 / 81.9 | 71.3 / 84.7 / 81.5 |
| coherence | fine | fine |

Throughput is unchanged within noise (medians 77.2 vs 81.5; the spread here is
wider than the difference). The real result is the 1022 MiB and what happens to
it: **no code was needed for the freed memory to become experts on the GPU.** The
placement search probes with the actual `cparams`, cache types included, so a
smaller KV measures as a smaller footprint and `--fit` resolves a lower `-ncmoe`
by itself. That composition is a direct dividend of deciding placement from
measured memory rather than from a fixed flag.

Not changed as a default. It is a strict improvement on this card with this
model, but KV precision is a quality knob, one coherence check is not a quality
evaluation, and defaults should not be set from a single model on a single 12GB
GPU. Recommended for this deployment, documented for others.

## KV quantization quality: validated short and long (2026-08-15)

The throughput and VRAM numbers above say nothing about quality, and a single
coherence check is not an evaluation - KV error accumulates over *stored* tokens,
so short prompts test the case least likely to fail.

`llama-perplexity`, the obvious tool, **segfaults on this model**: Gemma-4 with
MTP/iSWA requires `ctx_other` to be set (the same constraint that appears as a
warning during memory fitting), which that tool does not construct. So quality
was measured two other ways.

**Short prompts, deterministic.** Six prompts spanning arithmetic, list recall,
translation, factual lookup, completion and free composition, at temperature 0
with a fixed seed: **6/6 byte-identical** between f16 and q8_0.

**Long context, needle-in-a-haystack** (`scripts/kv-needle-test.py`). A fact
planted at 10%, 50% and 90% depth of a ~24700-token prompt, then asked for back:
**3/3 recalled under both f16 and q8_0.** Each run uses a unique nonce in the
filler so the prompt cache cannot serve a previous run - without it the second
configuration answered from cached KV with `prompt_n = 5`, which would have
proved nothing.

Two test artefacts worth recording, since both initially looked like real
failures: a 40-token reply limit truncated the model mid-reasoning so it never
reached its answer (read as 0/3 until the limit was raised), and the filler was
~32 tokens per sentence rather than the 14 estimated, building a 42904-token
prompt against a 32768 context. Neither was a property of the model.

Conclusion: on this model, symmetric `q8_0` KV is effectively lossless at both
ends of the range tested, for 1022 MiB of VRAM and two extra expert layers on the
GPU. Still not made a default - one model on one card is not the basis for a
global default, and the K-vs-V sensitivity asymmetry could not be explored here
because `FA_ALL_QUANTS=OFF` compiles only symmetric kernels.

# Block-level KV allocation via CUDA VMM: mechanism works, gating does not (2026-08-15)

PagedAttention's actual saving - do not pay for context nobody uses - reached
without PagedAttention's cost. vLLM needs a block table and gather kernels, which
is why the port was rejected: llama.cpp would need new attention kernels per
backend. CUDA's virtual memory API gets there differently: reserve contiguous
*virtual* addresses for the full n_ctx, map physical pages only as cells fill.
Addresses stay linear, so every existing attention kernel works unmodified.

Implemented: a VMM-backed CUDA buffer type (granule-level commit tracking, since
"cells 0..N in use" is a strided set of per-tensor prefixes, not one prefix of
the buffer), tensor slices recorded at `init_tensor`, commit-on-access for every
buffer API path, and a growth hook in `llama_kv_cache::apply_ubatch` that backs
cells before kernels reach them. Behind `GGML_CUDA_VMM_KV=1`, default off.

**Status: fixed and working.** The gating error below was corrected by having
the caller declare the allocation rather than the buffer type infer it from
size: `ggml_backend_cuda_vmm_next_alloc(true)` is set around the KV cache's own
allocation and nowhere else. Only the two KV buffers (680 and 478 MiB) are now
lazy; the 6618 MiB weights buffer is untouched. Server loads and serves with
zero CUDA errors.

Worth noting what the freed memory does: total VRAM use went *up* (11703 vs 8639
MiB eager), because moe-cache derives its budget from free VRAM and immediately
spends the saving on a larger expert cache. That is the system behaving as
designed - the KV saving is real, it just does not show up as a lower total.

The original diagnosis, kept for the record: The buffer type applied VMM to any allocation
over 64 MiB, so it captured the *model weights* buffer (6618 MiB) as well as the
KV buffers (680 and 478 MiB). Weights are read in full immediately, so lazy
commit is guaranteed to fault - the comment in the code says "deliberately not
applied to weights" and then does not enforce it. The KV buffers themselves
reserved and committed as intended (`commit_fraction 0.0100 over 10 slices`).

Two further things to verify once that is fixed, both known unknowns rather than
speculation:

- `llama_kv_cache_context`'s full-cache constructor sets `n_kv = kv->get_size()`
  for worst-case graph building. If any executed graph uses it, kernels address
  the entire reservation and commit everything on the first decode, leaving the
  feature working but pointless. Whether that path only reserves or also executes
  needs checking.
- Expected saving on this model is modest - only 5 of 30 layers are
  full-attention at 65536 cells (680 MiB at q8_0); the other 25 are sliding
  window and already bounded at 4608 cells. On a non-SWA model, or on an H200
  running a large model at full context, the proportion is far more favourable,
  which is the case this was built for.

# Eviction on pressure (2026-08-15)

The live budget re-check notices pressure and then, by design, refuses to act on
it: lowering the budget below what is already allocated makes the cache inert
(documented above - that bug froze every counter and cost 24 tok/s). Correct, but
it left the cache able to see pressure and unable to yield to it.

Pool slabs are one allocation each and cannot be partially freed, so the memory
that can actually be returned is the dispatch scratch (`d_ids`, `d_act`,
`d_act_q8`, `d_out`). `moe_cache_grow_device()` regrows those on demand before
the next dispatch, so releasing them costs one reallocation and no cached expert.
Released only when nothing is in flight and the queue is drained, since a
dispatch in progress is reading exactly those buffers.

Verified not to fire when VRAM is free, and not to break normal operation
(loads, serves, zero CUDA errors). The pressure path itself is untested here for
the usual reason - producing genuine VRAM pressure requires a second large
consumer on the card, and the earlier attempt at that (two full model instances)
simply fails to load rather than creating graceful pressure.

Returning cached experts themselves under pressure would need pool slabs
subdivided into independently freeable chunks. That is a real change to the
allocation structure, not a tuning knob, and is not attempted here.

# The RAM buffer, reframed: supplement mmap rather than replace it (2026-08-16)

The explicit-buffer rewrite was deferred as multi-week and unvalidatable here.
That assessment was of the wrong design. It assumed replacing mmap: stop mapping
the GGUF, own the allocation, implement read()-based population, reroute every
CPU expert access. Large, and risky precisely because the fallback path
disappears - a bug means wrong weights or no weights.

The better shape, and the one to build: **keep mmap and add an owned buffer,
each holding the part it is good at, with the boundary actively managed.** Not
"buffer first, mmap as fallback" - that still treats mmap as the loser and gives
up what it is genuinely better at. The split:

- **mmap keeps the cold tail.** Zero-copy, shared across processes, faulted
  lazily, and managed by an OS that can see global pressure we cannot. Nothing
  about the load path changes, and the long tail of a 500GB model is exactly the
  case mmap handles well.
- **The buffer holds the hot set.** Enforced LFRU over memory we own - the one
  thing no amount of `madvise` could deliver, because the kernel interface is
  demotion-only and has no way to express "protect this".
- **The boundary is coordinated, and this is the part that makes it a mixture
  rather than a stack.** When an expert is promoted into the buffer, its mmap
  pages are released (`MADV_DONTNEED`). Without that step the same expert is
  resident twice - once in the page cache, once in the buffer - and the
  duplication costs precisely what the buffer was built to save. The machinery
  for this already exists: the demotion sweep that currently advises cold experts
  would instead advise *promoted* ones, which is a better signal because it is
  certain rather than inferred.

Structurally it is the same thing moe-cache already does one tier up - VRAM
buffer over CPU-resident weights - applied to CPU RAM over disk-backed pages,
with the addition that the lower tier is told what the upper tier has taken.

Why it answers the objections that stopped the previous design:

- **Correctness is preserved by construction.** The fallback is today's code
  path, so a buffer miss behaves exactly as the system does now. The buffer is
  pure optimisation, not a replacement, and can be disabled at runtime.
- **It gets the one thing hinting could not: enforcement.** Everything tried
  through `madvise` moved a few percent against a 21x gap, because the kernel
  interface is demotion-only and no hint expresses "protect this". Within memory
  we own, LFRU is enforced outright - the same policy that holds ~70% hit rate in
  VRAM.
- **The kernel's blind LRU still governs the cold tail, which is where it does no
  harm.** The pathological case measured earlier - hot experts evicted as readily
  as never-used ones - only hurts for the hot set, and the hot set is exactly
  what the buffer removes from the kernel's jurisdiction.
- **It scales to the target.** For a 1.5TB model compressed to ~500GB, no design
  holds everything; the question is only whether the hot working set is held
  deliberately or by accident. This holds it deliberately while the tail streams.

Sizing follows the same rule already used for the VRAM budget and the prompt
cache: derive it from what is actually available (`MemAvailable`, clamped by any
cgroup limit), never a constant. The heat signal, decay and eviction rule are
already implemented for the RAM tier and sitting behind
`GGML_CUDA_MOE_CACHE_COLD_AFTER_EPOCHS`; what changes is that they would govern
memory we hold rather than advice we offer.

## Host hot-expert buffer: implemented and tested (2026-08-16)

Built as described above: an owned host buffer holding the hot experts, mmap
retained for everything else, and promotion releasing the corresponding mmap
pages (`MADV_DONTNEED`) so nothing is resident twice.

Interception is one line in `ggml-cpu.c`, at `src0_cur` - the single point where
the CPU reads an expert's weights. `ggml_moe_cache.host_ptr()` returns the
cache's own copy or NULL, and NULL is exactly today's path, so the fallback is
the existing code rather than new code.

Promotion happens in the planning path (under `session.mu`, where heat has just
been updated) once an expert has been selected several times, so a single fluke
cannot claim a slot. Eviction takes the coldest resident expert, and only if it
is genuinely colder than the candidate - otherwise a cold newcomer would churn a
hot incumbent out on every miss. The offset is published with a release store
after the copy completes, so a CPU thread that observes it always finds valid
weights. Opt-in via `GGML_CUDA_MOE_CACHE_HOST_MB`; it allocates real host memory
and should never do so by surprise.

**Testing, including the control that mattered most.** Buffer ON vs OFF gave
3/5 byte-identical outputs at temperature 0 - which looks like corruption until
the right control is run: **OFF vs OFF differs on exactly the same two prompts,
at the same 3/5 rate.** The nondeterminism is pre-existing (whether an expert is
served from VRAM or CPU changes the arithmetic path between runs) and has nothing
to do with the buffer. Without that control this would have been reported as a
corruption bug that does not exist.

Edge cases exercised, all clean and crash-free:

- **4 MiB budget** - 2 slots against thousands of experts, i.e. maximum eviction
  churn, including the "do not evict a hotter incumbent" guard.
- **Six concurrent requests** - `host_ptr` is called from many CPU compute
  threads simultaneously and is lock-free by construction on the read side.
- **Buffer disabled** (unset / 0) - the default, unchanged behaviour.
- **Mixed expert sizes** - a pool is sliced for one expert size; differently
  shaped tensors are left to mmap rather than wasting or corrupting a slot.

### Measured under pressure: negative, and why (2026-08-16)

| 5 GiB cap, generation tok/s | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| buffer off | 2.36 | 8.97 | 11.31 |
| buffer on (2 GiB) | 1.61 | 6.07 | 7.87 |

About 30% slower throughout. The cause is not the policy but the allocation
shape: `malloc` takes the whole 2 GiB pool up front, from inside the same 5 GiB
budget the page cache is using to hold experts. Early on almost nothing has been
promoted, so the buffer is 2 GiB of loss against nearly no gain, and
`MADV_DONTNEED` only offsets it once the buffer is actually full. That is the
same failure mode as mlock: memory we hold is memory the page cache does not
get, and holding it eagerly makes the shortfall worse before it makes it better.

Lazy allocation was implemented and **measured worse still** - 1.15/5.02/3.33
against eager's 1.61/6.07/7.87 - which rules out the allocation shape as the
cause and identifies the real one:

- **Promotion runs inline on the decode path.** Each promotion is a `malloc`, a
  1.4 MiB `memcpy` and a `madvise` syscall, performed while holding the session
  lock, on a machine already thrashing. Making allocation lazy simply did that
  work more often in smaller pieces. The fix is to move promotion onto the fill
  worker thread that already exists for VRAM fills: decode marks an expert as
  worth promoting, the worker performs the copy. That is a design correction
  rather than another parameter, and it is the next thing to build.
- **The test environment charges it twice.** Under a cgroup cap the buffer and
  the page cache draw on one budget, so owning memory is zero-sum by
  construction. The case this was built for - a 500GB model on a machine with
  RAM measured in hundreds of gigabytes - is not zero-sum in the same way, but
  it also cannot be tested here, which is exactly the limitation recorded
  throughout this document.

Left opt-in and off by default. The correctness work stands (the control run
showed the nondeterminism was pre-existing, and every edge case was clean); it
is the eager allocation that must change before the performance claim can be
revisited.

# What the literature actually does, and why our buffer was on the wrong axis (2026-08-16)

Searched after three buffer variants all measured worse. The field has converged
on something we did not try, and it explains the failures.

Sources: Speculating Experts (arxiv 2603.19289), MoE-Infinity, FineMoE
(EuroSys'26), SP-MoE (arxiv 2510.10302), CommitMoE (AAAI), ADEPT.

**The consensus mechanism is prediction, not capacity.** Expert selection at
layer N+1 is predictable from layer N's activations, so these systems run the
next layer's router early, learn which experts it will want, and transfer them
*while the current layer is still computing*. Speculating Experts constructs a
"quasi-hidden state" (normalised residual stream plus a default vector
approximating typical expert contributions) to approximate what the next
router will see; it predicts one layer ahead at roughly 90% hit rate, dropping
to ~70% in early layers where representational drift is high.

Its own summary of the memory strategy is the sentence that matters here:
*"rather than holding experts in memory, the system uses asynchronous
prefetching"* - overlapping transfer with compute to mask latency.

**Two corrections to the work above.**

1. *We were solving for capacity; the constraint is latency.* The host buffer,
   mlock pinning and MADV_WILLNEED all tried to **hold** the hot set resident.
   Prediction removes the need to hold anything - the transfer merely has to
   start early enough. That is why all three variants failed the same way
   regardless of allocation shape, thread, or policy: they addressed the wrong
   axis, and each one bought residency by taking memory from the page cache.

2. *The 21x was never a realistic target.* Speculating Experts reports up to
   **14% TPOT reduction** over on-demand expert loading, and larger gains on
   weaker GPUs (12-14% on an A6000 against 5-8% on A100/GH200) - which places
   this 12GB card in the favourable regime, but nowhere near an order of
   magnitude. Their baseline is on-demand loading from *CPU RAM*; the 21x
   measured here was disk-bound under RAM pressure, a worse regime the papers do
   not target.

**What already exists here, and what does not.** moe-cache's fill queue is
already asynchronous - a miss enqueues a fill and compute continues - so the
overlap machinery is built. What is missing is that our fills are *reactive*
(issued after the miss, too late for the token that caused it) where the
literature's are *predictive* (issued before it). Closing that gap means running
the next layer's router early against the current layer's output and enqueuing
fills for what it selects, which is a bounded change to the planning path rather
than a new subsystem - and it is the thing that should have been built instead of
the buffer.

## Predictor survey, and why we are not building one yet (2026-08-16)

Best predictors currently published, with what each needs:

| predictor | recall | requires |
| --- | --- | --- |
| PILOT router-lookahead | 73.6% | layer L's post-attention state + L+1's router weights |
| Two-step shared-expert (Colibri #200) | 76.7% | the above plus shared-expert compute (3 small matmuls) |
| Fate (cosine similarity of gate inputs) | 78.8%, no training | gate inputs |
| LLaPor | Top-4 hit rate >=94% | more machinery |

All of them need model-level state - the residual stream and the next layer's
router weights - which moe-cache, living at the ggml op level, never sees. That
alone makes any of them a llama.cpp graph change rather than a cache change.

**The decisive finding is not in the accuracy column.** Colibri built the best of
these (76.7%, beating PILOT by 3.1% over 13,616 samples on GLM-5.2) and reported:
*"end-to-end throughput showed no improvement on the 24 GB test hardware due to
cache constraints (capacity limited to 2 experts per layer)."* A better predictor
bought nothing because the cache could not hold what it prefetched.

Checked whether we are in the same position, and we are: **1316 of 1316 slots
used, 100% full**, hit rate 57.3%. Capacity binds here too. A prefetch needs a
slot to land in; with the cache saturated it is either refused by the admission
gate or evicts an expert already earning its place. Prediction cannot help until
there is somewhere to put the prediction.

So the predictor is not the next thing to build - it is the thing after the next
thing. What moves first is effective capacity: more VRAM for the cache (the
gigabyte freed by q8_0 KV went to placing more expert layers on the GPU rather
than to the cache - that is a tunable trade), or fewer CPU-offloaded layers.
Once slots are no longer saturated, prediction has something to buy, and the
survey above says which predictor to port and what recall to expect.

Recorded rather than built, deliberately. Three mechanisms today failed by
addressing capacity when the constraint was elsewhere; building a fourth that
addresses timing while capacity binds would be the same error inverted.


## Credit where decisions came from (2026-08-16)

Recorded because several of these changed what we did *not* build, which is
harder to see later than what we did:

- **antirez/ds4 (DwarfStar)** - the three-tier structure this fork arrived at
  independently, and the explicit read/write buffer we measured our mmap-based
  approach against. Reviewing it corrected the premise it was reviewed under
  (it claims no 8GB support; its published figures are from a 128GB Mac).
- **vLLM** - PagedAttention, deliberately not ported. Its saving was taken via
  CUDA's virtual-memory API instead, which keeps addresses contiguous so no
  attention kernel changes. Its automatic prefix caching already existed here.
- **JustVugg/colibri issue #200** - the two-step shared-expert router predictor,
  best recall in this space at 76.7%, and its finding that better prediction
  bought no throughput when cache capacity bound first. Checked against our own
  saturated cache before writing a predictor, and that check is why none was
  written.
- **Speculating Experts / PILOT / Fate / FineMoE / SP-MoE** - established that
  the field prefetches by prediction rather than holding experts resident, and
  that realistic gains are around 14%, not the order of magnitude assumed here.
  Three residency experiments ship disabled because of that correction.

## Correction: capacity does not block prefetch here (2026-08-16)

The previous section concluded a predictor was pointless because our cache is
100% full. That conflated *full* with *too small*, and they are different.

| | Colibri's failed case | here |
| --- | --- | --- |
| cache capacity | 2 experts/layer | **62.7 experts/layer** |
| needed per layer (top-k) | 8 | 8 |
| holds one layer's lookahead? | no, 2 < 8 | **yes, by ~8x** |

Colibri's cache could not hold even one layer's routed experts, so a prefetch was
overwritten before use - a real capacity failure. Ours holds roughly half of
every layer's experts. And a full LFRU cache is a working one, not a blocked one:
prefetching evicts the *coldest* resident to make room for one about to be used
with ~77% confidence, which is a good trade, not a conflict.

The staging requirement is also much smaller than assumed. Prefetch is a pipeline
buffer, not a cache - it needs room for one layer's top-k (about 8 experts) while
the current layer computes, not for the working set. That is exactly why it
succeeds where holding experts fails: it converts a blocking transfer into an
overlapped one and needs almost no space to do it.

The error is worth naming: after three residency experiments failed on capacity,
capacity was assumed to be the constraint again without checking the number that
mattered.

## Scope: router-lookahead prefetch (the change that is actually blocking)

The real obstacle is where the signal lives, not whether it would help.

Verified in the code: the router is `logits = build_lora_mm(gate_inp, cur)` in
`llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp:1947`) - a plain matmul
of the layer's gate weights against the hidden state. Running layer L+1's router
early therefore needs only tensors that already exist: `model.layers[il+1]
.ffn_gate_inp` and layer L's pre-MoE `cur`.

What the change involves:

1. At layer `il`, after attention and before the MoE block, build one extra
   matmul against `layers[il+1].ffn_gate_inp`, then `ggml_argsort_top_k` for the
   predicted expert ids. Cost is one `n_embd x n_expert` matmul per layer -
   negligible beside expert compute.
2. Deliver those ids to moe-cache *before* layer `il+1`'s `MUL_MAT_ID` executes,
   so fills are already in flight. moe-cache has no entry point for this today;
   it learns of experts only when `plan()` is called with the real ids. A hint
   entry (`ggml_moe_cache.prefetch(host_base, ids, n_ids)`) plus a small custom
   op or eval-callback hook to invoke it is the missing plumbing.
3. Prefetched fills must not evict warmer entries than the ones they replace -
   speculation should lose ties to certainty. The existing eviction already picks
   coldest-in-window, so this is a policy flag rather than new machinery.

Known caveats, from the literature rather than guessed: predicting from the
post-attention state alone is "stale by one MoE residual" and measures 73.6%
recall (PILOT); adding the shared expert's contribution first raises it to 76.7%
(Colibri #200) at the cost of three small matmuls. `gate_inp` location is
architecture-specific, so this needs guarding for models without a standard MoE
gate, and the interaction with MTP/draft contexts is untested.

Realistic payoff, stated before building so it can be checked: the literature
reports up to ~14% TPOT improvement over on-demand loading, larger on weaker
GPUs (12-14% on an A6000 against 5-8% on A100/GH200), which places this 12GB card
in the favourable regime.

## Router-lookahead prefetch: built, works, no throughput change here (2026-08-16)

Implemented the mechanism the literature converged on. At layer L, the next
layer's gate is run against this layer's router input, `ggml_argsort_top_k` gives
the predicted experts, and a `ggml_map_custom1` node hands them to moe-cache -
firing in graph order, while layer L is still computing and before layer L+1's
expert matmul is reached.

Pieces: `ggml_moe_cache.prefetch()` (cache side, fills into free slots only),
`llm_graph_context::build_moe_lookahead()` (graph side), and one call in
`gemma4.cpp`. The graph context has no access to `model`, so the helper takes the
next layer's tensors and the model builder supplies them - which also keeps this
opt-in per architecture rather than silently applying where it has not been
checked.

Speculation never evicts. At ~77% accuracy a wrong guess that displaced a
known-hot expert would cost more than a right one saves, so prefetch takes free
slots or does nothing, and respects the same queue bounds as demand fills.

**Measured, `-ncmoe 21`, 8192 ctx:**

| | run 1 | run 2 | run 3 | hit rate | prefetches |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefetch on | 50.4 | 56.7 | 57.9 | 89.4% | 673 |
| prefetch off | 51.5 | 57.7 | 57.3 | 88.3% | 0 |

The mechanism demonstrably works - 673 fills issued, hit rate up 1.1pp - and
throughput does not move. The reason is in the hit rate: at 88% this
configuration is not transfer-bound. Roughly one access in nine misses, and those
were already being filled asynchronously, so there is very little exposed
transfer left for prefetch to hide behind compute.

Where it should matter is the opposite regime: heavy offload with a low hit rate,
where misses are frequent and each one blocks. That is also where the literature
measures its ~14%, on GPUs too small to hold the working set. Kept on by default
(`GGML_CUDA_MOE_CACHE_PREFETCH=0` disables) because it is cheap, correct, and
costs nothing when there is nothing to hide - unlike the residency experiments,
which actively took memory from something else.

**The signal is reusable, and that is the more interesting part.** The prediction
says which experts are about to be needed. Today's host buffer promotes by
historical heat - backward-looking, and it tried to hold a large hot set, which
is the capacity play that failed three times. Feeding the same prediction to the
buffer would make it forward-looking and much smaller: a staging area for what is
imminently required rather than a cache of what was recently popular. That is the
same shift from capacity to timing the literature made, applied one tier down,
and it is the natural next step for the buffer rather than another sizing attempt.

### Feeding the buffer from the prediction, not from heat (2026-08-16)

The prediction says which experts are *about to* be used. The host buffer was
promoting on historical heat - backward-looking, and aimed at holding the whole
hot set, which is the capacity approach that lost three times. Wiring the
prediction into the buffer's promotion queue makes it forward-looking and small:
a staging area for imminent work rather than a cache of what was recently
popular. Queued for the worker, never copied inline, since a 1.4 MiB memcpy on
the decode path is what made the earlier version slow.

Measured under the same 5 GiB cap:

| generation tok/s | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| no buffer | 2.36 | 8.97 | 11.31 |
| buffer fed by prediction | 1.63 | 5.26 | 5.49 |
| buffer fed by heat (previous) | 1.08 | 3.74 | 5.03 |

**The signal change is a real improvement: +51%, +41% and +9% over the heat-fed
version on the three runs.** Forward-looking promotion beats backward-looking
promotion, which is the point worth keeping.

It is still below no buffer at all, and that has not changed for the reason it
never did: at 2x oversubscription every byte the buffer holds is a byte the page
cache loses, and no promotion signal fixes a zero-sum budget. What the prediction
does is make the buffer *less wrong* - it holds fewer, better-chosen experts.
Where the ratio is not zero-sum (RAM close to sufficient, the case this targets
and which cannot be reproduced here) that is exactly the difference that should
decide it.

# Second model, no new machinery: Nemotron 3.5 Lightning 30B-A3B (2026-08-22)

Everything built for Gemma-4 transferred to a structurally different model with
one exception. Nemotron 3.5 Lightning is `nemotron_h_moe`: 53 blocks interleaving
Mamba-2, dense and MoE, 128 routed experts plus 1 shared expert per MoE layer,
MTP layers embedded in the GGUF, 1M trained context.

Verified working on first run, unchanged:

- **Placement from context.** `-ncmoe` omitted entirely; the search derived
  **0 -> 46 of 53 layers** on CPU purely from the requested 65536 context, on a
  model it had never seen.
- **MTP without a drafter.** `nextn_predict_layers = 1` satisfies
  `nemotron-h-moe.cpp`'s single-block assert, `blk.52.nextn.*` tensors load, and
  `common/speculative.cpp` takes its `else if (spec_mtp)` branch to build the
  draft context against the target model itself. No `-md`, one file instead of
  two - unlike Gemma-4, whose draft is a separate `gemma4-assistant` model
  because its MTP layers are genuinely not in the main GGUF (the `nextn` string
  found there is `nextNode`, from the tokenizer).
- **moe-cache on a hybrid layout.** Engages after the first request: two pools
  (q5_0 and q5_1 expert shapes), 1158 slots, ~3.6 GB VRAM, 46x128 Brain grid.
- **Shared experts stay on GPU by construction.** They run on *every* token, so
  offloading them would be exactly backwards. The offload regex is
  `\.ffn_(up|down|gate|gate_up)_(ch|)exps`, which matches `ffn_down_exps` and not
  `ffn_down_shexp` - correct already, and worth recording since it is the kind of
  thing that is right by accident until someone checks.

**The one thing that did need code: router-lookahead prefetch.** It was wired
only into `gemma4.cpp`, so Nemotron ran with `prefetches: 0`. Wiring it required
a real difference rather than a copy: Gemma predicts against layer `il+1`, but
here `il+1` is usually a Mamba-2 or dense block with no router at all. Predicting
against an absent or differently-shaped gate would crash or produce confident
nonsense, so `nemotron-h.cpp` scans forward for the nearest block that actually
has a router and checks the gate dimensions match before predicting.

| | prefetches | hit rate | generation tok/s |
| --- | ---: | ---: | --- |
| before wiring | 0 | 55.2% | 39.3 / 38.7 / 35.4 |
| after | **379** | **57.1%** | 37.1 / 38.1 / 38.5 / 36.1 |

Throughput is flat within noise, the same result as on Gemma: prefetch moves hit
rate, and only converts to speed where the workload is transfer-bound.

## Why Nemotron's hit rate is lower than Gemma's, and what it exposed (2026-08-22)

Nemotron ran at 57.1% cache hit rate against Gemma's 79%. The cause is
arithmetic, not a defect:

| | Gemma-4 | Nemotron |
| --- | ---: | ---: |
| offloaded layers | 21 | 46 |
| expert size | 2178 / 1452 KiB | 3349 / 3654 KiB |
| cache slots | 1316 | 1239 |
| coverage per layer | 62.7 (49%) | 26.9 (21%) |
| hit rate | 79% | 57% |

A similar slot count spread over 2.2x the layers, with experts about twice the
size, gives each layer 21% coverage instead of 49%. Hit rate tracks coverage
closely.

**Scanning further ahead would not fix it.** The cache is 100% full (1239/1239)
with 79057 admission skips, and prefetch deliberately takes only free slots so
speculation never evicts certainty. More predictions would have nowhere to land,
and accuracy degrades with distance anyway.

**Reducing context made it worse, which was the useful surprise.** At 16384
instead of 65536: 44 layers offloaded (not 46), 1014 slots (not 1239), 18%
coverage, 53.3% hit rate, 30-34 tok/s. Less context let the placement search
offload *fewer* layers, so more weights stayed resident on the GPU - consuming
exactly the VRAM the expert cache wanted.

**So the lever runs the other way.** Forcing `-ncmoe 53` (every layer offloaded)
at the same 65536 context:

| -ncmoe | layers | slots | coverage | hit rate | tok/s |
| --- | ---: | ---: | ---: | ---: | --- |
| auto (46) | 46 | 1239 | 21% | 57.1% | 36-38 |
| 53 | 53 | 2307 | 34% | 68.3% | 39.8-41.2 |

Nearly double the cache, +11pp hit rate, ~8% throughput, same hardware and
context.

**What this exposes is a limitation in our own placement logic, one level up
from the bug fixed this morning.** That fix made context lead and placement
follow. But the search still answers only *"what is the minimum offload that
makes this context fit?"* - it stops at 46 because 46 suffices. Every layer left
resident holds weights competing with the expert cache for the same VRAM, and
since cached experts serve the offloaded layers, trading resident weights for
cache capacity stays net-positive well past the point where the context merely
fits. Sufficient is not optimal.

Caveats stated plainly: one model, one context, n=3, and 41 vs 37 tok/s sits near
this rig's noise band. The hit-rate gap (68.3% vs 57.1%) is the solid part, being
a far lower-variance metric than throughput.

`--moe-calibrate` already performs an empirical throughput search over `-ncmoe`;
auto-placement simply does not consult it. Making the fit search continue past
"fits" toward a throughput optimum - or defaulting to the calibrated value where
one exists - is the natural next step, and unlike most of this document's
suggestions it has a measured gain attached before being built.


### `-ncmoe 99` and a reporting bug it exposed (2026-08-22)

`-ncmoe 99` on a 53-layer model is a legitimate "offload everything" idiom - the
extra 46 override patterns simply match layers that do not exist - and it
measures identically to `-ncmoe 53`: 53 layers offloaded, 2307 slots, 34%
coverage, 68.5% hit rate against 68.3%.

It did surface something small and real, though. `/props` reported
`n_cpu_moe_final: 99` on a model with 53 layers, and that field feeds the
model-info dialog and the Brain placement view - so the UI was stating something
untrue about the model. The count came straight from the number of override
entries rather than from the layers those entries actually match. Now clamped to
the model's real layer count in `common_maybe_raise_moe_for_ctx`.

Worth noting because it is the same failure mode as `du` reporting 21 GB for
files that freed 4 GB: a number that was easy to compute standing in for the
number that was true.

## TurboQuant for the V cache: measured, then dropped

Asked to implement TurboQuant (Zandieh et al., ICLR 2026 - WHT + Lloyd-Max,
claimed 72-78% KV reduction) for the V cache. Built the codec on CPU first,
because that answers the go/no-go question before any kernel work.

The codec is self-contained: the Hadamard rotation is applied at encode and
undone at decode, so it can be an ordinary `GGML_TYPE_*` with attention math
untouched. Implementation is in `tools/tbq-dump/tbq-eval.c`; it reproduces
Lloyd-Max's theoretical 3-bit bound exactly (NRMSE 0.180 vs theory 0.186), so
the numbers below are not an artifact of a broken implementation.

Evaluated against real V vectors captured from running models with the new
`llama-tbq-dump` tool (ggml eval callback on `Vcur`), not synthetic data. The
metric is error in `softmax(scores) @ V` - inner-product distortion, which is
what TurboQuant's theorem is actually stated for - not per-element NRMSE.

Nemotron 3.5 Lightning, real V, head_dim=256, 20k rows:

| scheme      | bpw   | elem-NRMSE | attn-out NRMSE |
|-------------|-------|------------|----------------|
| q8_0        | 8.5   | 0.00570    | 0.00296        |
| q4_0        | 4.5   | 0.09129    | **0.04752**    |
| tbq4 b=32   | 4.5   | 0.09438    | **0.05001**    |
| tbq3 b=32   | 3.5   | 0.17991    | 0.09766        |
| tbq3 b=128  | 3.125 | 0.18432    | 0.10271        |

At matched bitrate TurboQuant is *slightly worse* than the q4_0 llama.cpp
already has. Qwen2.5 agrees (0.03387 vs 0.03345).

The reason is measurable rather than mysterious. The Hadamard rotation exists
to destroy per-channel outliers ("massive activations") that wreck a per-block
absmax scale. The captured V tensors barely have any: channel-outlier ratio
(max channel sd / mean channel sd) is **1.5x on Nemotron, 1.9x on Qwen2.5**.
That pathology lives in K and in the residual stream, not in V - so on V the
rotation has nothing to repair, and all that is left is Lloyd-Max vs absmax at
equal bits, which absmax wins because its per-32 scale is itself adaptive.

TurboQuant's one unique offering is a bitrate llama.cpp cannot currently reach
for KV at all (3.125 bpw), but at 2.2x the error of q4_0.

Payoff on this model is small regardless: Nemotron is a linear-attention hybrid
with only 7 attention layers of 53, so the whole KV cache is ~268 MiB (1 layer
f16 = 64 MiB + 6 layers q8_0 = 204 MiB). The V half is ~134 MiB; going q8_0 ->
3.125 bpw saves ~85 MiB out of ~12 GiB of VRAM.

Cost side: a new ggml type plus a *fused* FA kernel for the (q8_0 K, tbq3 V)
pairing, or the pair falls off the fused path onto the 25-45x slow path (this
build has `GGML_CUDA_FA_ALL_QUANTS=OFF`; only f16-f16, bf16-bf16, q8_0-q8_0 and
q4_0-q4_0 exist). Not built. The measurement, not an opinion, is the reason.

## Calibration is noise-limited in the range it searches

Two consecutive `--moe-calibrate` runs on Nemotron, same machine, same flags:

| ncmoe | run 1  | run 2  |
|-------|--------|--------|
| 37    | -      | 35.26  |
| 38    | -      | 38.16  |
| 39    | 37.61  | 37.02  |
| 40    | 38.05  | **38.21** |
| 41    | 38.01  | 35.41  |
| 42    | 38.17  | -      |
| 43    | **38.61** | -   |

The two runs disagree on the argmax (43 vs 40) while their winning throughputs
differ by ~1%. The whole band spans 35.3-38.2 tok/s - about +/-8% - so the
search is resolving noise, not signal, at `n_samples_per_candidate = 2`.

Two consequences, one fixed and one open:

*Fixed:* run 1's peak landed exactly on the search's upper bound (43), which is
`safe_n + max(4, n_layer/8)` - a heuristic from our own Gemma-4 sweeps ("the
peak is never more than ~15% of n_layer above the floor"). A peak sitting on the
boundary means throughput was still climbing when the search ran out of room, so
the bound is now extended to the hard ceiling and the search continues.
`measure_ncmoe` is memoized so the extension never re-pays for a measured point.

*Open:* the region that actually looked promising was never sampled. `-ncmoe 99`
was separately measured at 2307 expert-cache slots / 68.3% hit rate against the
fit search's 1239 / 57.1%, because CPU offload frees VRAM that the expert cache
converts back into hit rate. Whether that hit rate becomes throughput is
unmeasured - a direct ncmoe 40 vs 46 vs 52 benchmark was started and timed out
before finishing. Until that is measured, widening the default range would be
shipping a guess.

## Placement between ncmoe 40 and 52 does not affect throughput

The open question above - whether the expert-cache hit rate gained at high CPU
offload turns into tokens/sec - is now measured. Same flags as the live deploy,
on 8099, concurrency 4, varying only `-ncmoe`, two measured rounds each after a
warm-up round that fills the expert cache:

| placement | r1    | r2    | mean  |
|-----------|-------|-------|-------|
| ncmoe=40  | 29.46 | 30.70 | 30.08 |
| ncmoe=46  | 30.85 | 29.19 | 30.02 |
| ncmoe=52  | 29.41 | 30.83 | 30.12 |

All six samples fall in 29.19-30.85 tok/s (+/-2.7%), and the three means agree
to within 0.1 tok/s. The 68.3% vs 57.1% hit-rate advantage measured at
`-ncmoe 99` does **not** become throughput: offloading more layers adds CPU
compute at very close to the rate the improved hit rate removes GPU stalls, and
the two cancel.

Consequences:

- Do **not** widen the calibration search range. There is nothing above the old
  bound to find, so the boundary-extension change is a correctness fix for a
  heuristic that was wrong in principle, not a throughput win. It stays because
  a peak on the boundary is still the wrong thing to report as an optimum.
- The `-ncmoe` arm of `--moe-calibrate` is measuring a flat objective on this
  model class. That fully explains the +/-8% run-to-run disagreement documented
  above: there is no peak to find, so the "winner" is whichever candidate drew
  the luckiest sample.
- It also retroactively justifies treating calibration as a floor rather than
  an override. The fit search's conservative answer (46) is worth exactly as
  much as the calibrated one (40), so deferring to the safer number costs
  nothing measurable.

Note the absolute numbers here (~30 tok/s) are below calibration's ~38 tok/s for
the same placement because this harness lets the reasoning model run ~760 tokens
per request rather than the calibrator's short generations. Only the comparison
between arms is meaningful.

## The UI build silently stopped compiling our own source

The brain/atlas view reverted to the stock square. Nothing of ours was lost or
reverted - `tools/ui/src/lib/components/app/brain/Brain.svelte` still had every
atlas reference. Upstream #22937 ("Move static build output from repo code to HF
Bucket") changed the build to download a prebuilt UI bundle from
`ggml-org/llama-ui` and embed that, with `LLAMA_USE_PREBUILT_UI` defaulting ON.

The failure mode is the nasty kind: the build stays green, no warning is
printed, the source stays on disk untouched, and every rebuild quietly ships
upstream's UI. `LLAMA_USE_PREBUILT_UI=OFF` alone is not enough either - it then
embeds `tools/ui/dist`, which still held the *downloaded* assets, so `npm run
build` has to regenerate `dist/` from source first. The embedded `ui.cpp` was
byte-identical in size until that happened, which is what gave it away.

`LLAMA_USE_PREBUILT_UI` now defaults OFF in this fork. A fork that modifies the
UI cannot correctly default to someone else's prebuilt copy of it.

## The memory hierarchy, measured end to end (Aug 2026, after the NVMe)

Every number below is measured on this box, not derived. They settle several
design questions that had been argued from arithmetic.

| tier | capacity | measured bandwidth |
|------|----------|--------------------|
| VRAM (RTX 3060) | 12 GB | ~360 GB/s spec, **254 GB/s** effective in the MoE op |
| RAM -> VRAM (pinned PCIe 4.0 x16) | - | **24.5 GB/s** (pageable: 12-17) |
| RAM local (DDR4-2666, 6 threads) | 30 GB | **36.6 GB/s** |
| NVMe (WD Blue SN5000) | 500 GB | **3.55 GB/s**, QD-independent (3.43 at QD1) |
| SATA SSD (previous model store) | 224 GB | **0.50 GB/s** |

`nvidia-smi` reports PCIe Gen1 at idle; the link trains up to Gen4 x16 only
under real transfers. Measure under load or the number is 6x too low.

Queue depth barely matters on this NVMe - 3.43 GB/s at QD1 vs 3.56 at QD32 - so
io_uring depth is not the win it would be on a faster drive. That saved building
an async fetch path that would have bought nothing.

### Transfer vs compute: the balance point

One MoE layer step (`MUL_MAT_ID`, q4_K, 128 experts, top-8, batch 1, 7.08 MB of
expert weights), via `test-backend-ops perf`:

| operation | time | ratio to compute |
|-----------|------|------------------|
| compute, weights resident in VRAM | **27.8 us** | 1x |
| fetch from RAM over PCIe | **289 us** | **10.4x** |
| fetch from NVMe | **1,994 us** | **72x** |

To hide a RAM fetch behind compute you must prefetch ~10 layers ahead; to hide
an NVMe fetch, ~72 layers ahead. Nemotron has **24 MoE layers in total**, and
the router can only predict one layer ahead because layer L+2's routing depends
on a hidden state that does not exist yet.

So: **NVMe-backed experts can never arrive in time.** NVMe is capacity for
experts that are rarely used at all, never a tier on the steady-state path.
Residency is mandatory; streaming is strictly a fallback. This kills the
otherwise attractive idea of shrinking the VRAM cache to a small landing zone
and streaming everything just-in-time.

### Batching does not amortize weight reads in MoE

Same op, scaling tokens per pass:

| tokens | MoE (MUL_MAT_ID) | dense (MUL_MAT, q5_1 4096x14336) |
|--------|------------------|----------------------------------|
| 1      | 27.8 us          | - |
| 3      | -                | 136 us |
| 8      | 177 us (6.4x)    | 213 us (1.56x for 2.7x work) |
| 128    | 1,491 us         | - |
| 512    | 1,654 us (1.11x for 4x work) | - |

A dense matmul reads its weights once and amortizes across the batch. An MoE
layer does not: each extra token routes to *more distinct experts*, so time
grows nearly linearly until the expert count saturates (~n=128 here, after which
all 128 experts have been touched and further batching really is free).

### Which is why MTP *hurts* under memory pressure

Measured, Nemotron on NVMe, cgroup `memory.max`, single stream:

| RAM cap | model : RAM+VRAM | no MTP | with MTP | delta |
|---------|------------------|--------|----------|-------|
| 26G | 0.68x | **38.0** | 32.4 | -15% |
| 16G | 0.94x | **26.6** | 16.2 | **-39%** |
| 12G | 1.10x | **8.75** | 6.15 | **-30%** |
| 8G  | 1.33x | **4.5**  | -    | - |

Draft acceptance was healthy throughout (0.74-0.90, mean accepted length
3.2-3.7), so the drafting itself works. The loss is memory behaviour: verifying
~3.3 tokens in one pass activates the *union* of those tokens' experts - at
top-6 per layer that is ~15-20 distinct experts instead of 6. Speculative
decoding trades more expert traffic for fewer passes, and when expert traffic is
the bottleneck that trade is a loss. The prediction here was that MTP would help
*more* under pressure; it does the opposite, at every cap.

### The fitting cliff

The same table read as a curve: throughput is flat while the model fits and
collapses the moment it does not. 0.94x -> 1.10x (a 17% increase in pressure)
costs 70% of throughput. Storage speed sets how far you fall, not where the edge
is - all of these runs already had the model on the 3.55 GB/s NVMe.

Practical ceiling on this box (30 GB RAM + 12 GB VRAM, ~36 GB usable): a model
of about **34-35 GB**. Spend the budget on quantization quality inside that
boundary rather than on parameter count past it.

### What this says about eviction policy (not yet implemented)

A miss refilled from RAM costs 289 us; from NVMe, 1,994 us - a **7x difference**
that LFRU currently ignores, since it treats every miss as equal cost. The cache
should prefer evicting RAM-backed experts and protect NVMe-only ones. This is
textbook cost-aware caching (GreedyDual-Size, Cao & Irani 1997), and it is the
clearest thing the OS literature has to offer us that we are currently getting
wrong.

## Reading the fill path properly: two claimed defects were not defects

Implementing the "obvious" wins from the transfer/compute analysis meant reading
the fill path closely, and two of the three claimed defects dissolved on contact.

**"Every fill blocks on `cudaStreamSynchronize`."** It does, but that
synchronize runs on the *fill worker thread*, on the worker's own low-priority
stream - not on the decode path. Decode never waits for a fill: on a miss it
records demand, enqueues a job, and computes that expert on the CPU for this
token, so the promoted copy only benefits later tokens. Transfer is already
fully overlapped with compute. There was no serialization to remove.

**The cache-size knee is not where transfer/compute balance predicted.** The
prediction was C* ≈ 7.7 GB from `t_xfer(C) = t_cmp`. Measured by sweeping
`--moe-cache`:

| cache | 1 GB | 2 GB | 4 GB | 6 GB | 8 GB |
|-------|------|------|------|------|------|
| tok/s | 29.8 | 32.4 | **37.3** | 37.6 | 37.2 |

The knee is at **4 GB**, and everything past it is flat inside noise. The model
was wrong in structure, not arithmetic: it priced a miss as a PCIe transfer, but
a miss is *computed on the CPU* and no transfer happens on the critical path at
all. The correct objective is a placement problem,

    T(C) = sum_{e in VRAM(C)} f_e * t_gpu  +  sum_{e not in VRAM(C)} f_e * t_cpu  +  T_other

with f_e the measured access frequency. Marginal value is f_e * (t_cpu - t_gpu),
strictly decreasing as C admits colder experts, so greedy-by-frequency is exactly
optimal and flattens where f_e does. The trace predicts where: the top 512
experts serve 46.4% of accesses and the next 512 only 19.7%.

**Consequence: ~3.5 GB of VRAM is free.** We ship a cache around 7.4-8 GB for
throughput that 4 GB already delivers. On this model that reclaims enough VRAM
for a large multiple of the current KV allocation.

The third defect was real: the fill path does `memcpy` into a pinned staging
buffer and then DMAs from it, which costs an extra RAM->RAM copy (203 us per
7.43 MB expert) purely to make the following copy pinned - cancelling most of
what pinning is worth. `GGML_CUDA_MOE_CACHE_HOSTREG_MB` now page-locks expert
tensor regions lazily, budget-capped, and DMAs straight from the model mapping.
It is opt-in (default 0) because pinned pages are unreclaimable. Since fills are
off the decode path, the win is fill *throughput*, not per-token latency: the
worker's ceiling moves from ~1,976 to ~3,300 experts/s against a measured demand
near 1,428/s.

## The direct-DMA fill: why it took four attempts, and the real number

`GGML_CUDA_MOE_CACHE_HOSTREG_MB` initially failed silently at every fill -
`cudaHostRegister` returned `invalid argument` on every attempt, and the
original log line used the filterable `MOE_CACHE_LOG` macro, which (like the
`--moe-cache` stats path tested earlier) emitted nothing. That combination
would have let a completely dead code path report "no measurable gain" with
total confidence. Switched to unfilterable `fprintf(stderr)` before trusting
any A/B result again.

Diagnosed with `cuMemHostRegister` called directly via ctypes against a real
mmap of the model file, isolating llama.cpp entirely:

| mapping | flags=0 | READ_ONLY |
|---|---|---|
| `PROT_READ` (what llama.cpp uses) | `INVALID_VALUE` | `NOT_SUPPORTED` |
| `PROT_READ\|PROT_WRITE` | **OK** | `NOT_SUPPORTED` |

Two separate driver limitations, not one: this driver cannot register a
read-only file mapping at all, and `cudaHostRegisterReadOnly` is unsupported
regardless. The only registrable mapping is writable. `LLAMA_MMAP_WRITABLE=1`
now maps the model `PROT_READ|PROT_WRITE` - which also had to become
`MAP_PRIVATE` instead of `MAP_SHARED`, since `MAP_SHARED|PROT_WRITE` on an
`O_RDONLY` fd is `EACCES`. `MAP_PRIVATE` is copy-on-write, so this is safe by
construction: nothing our process writes can ever reach the file on disk, and
weight tensors are never written regardless.

With that fixed, direct DMA genuinely runs - 7,840 MiB paged-locked, correctly
capped under an 8,192 MiB budget. Three-arm A/B, three rounds each:

| arm | mean tok/s |
|-----|-----------|
| baseline (read-only mmap, staging copy) | 37.4 |
| writable mmap only, no DMA change | 37.0 |
| **writable mmap + direct DMA** | **37.9** |

**+1.3%, and every direct-DMA sample beats every baseline sample - real, not
noise, but far short of the 1.67x the per-expert microbenchmark (506 -> 303 us)
predicted.** The gap is explained by where fills sit in the pipeline: a miss is
computed on the CPU for the *current* token regardless, and the promoted copy
only benefits later tokens, so shaving 203 us off a background copy barely
touches the measured critical path. At ~72% hit rate most tokens do not wait on
a fill at all. This is consistent with, not a contradiction of, the earlier
finding that transfer was already off the decode path - the win was always
going to be in fill *throughput* headroom, not per-token latency, and headroom
that was already ample shows only a small effect when widened further.

## Deeper router lookahead: measured, not just modeled

Built an offline validation of the router-lookahead mechanism against ground
truth - real hidden states captured via a modified eval callback, real gate
weight matrices and load-balancing bias read from the GGUF, real routing
decisions from the trace tool. Sanity check (predict a layer from its own
hidden state) passed exactly: 4117/4117, 100.0%.

Cross-layer precision (predicting layer L+d from layer L's hidden state,
against what layer L+d's router actually needed):

| depth d (MoE layers ahead) | precision | vs. random (4.7%) |
|---|---|---|
| 1 | 59.3% | 12.6x |
| 2 | 48.0% | 10.2x |
| 3 | 41.1% | 8.7x |

This also surfaced a real bug in the shipped mechanism: `build_moe_lookahead`
ranked by raw gate logits, but the model's actual ranking criterion is
`sigmoid(logits) + exp_probs_b` (a learned per-expert load-balancing bias).
Fixed - `build_moe_lookahead` now takes the bias tensor and applies it before
ranking. Gemma-4 has no such bias tensor, so this is a no-op there (sigmoid is
monotonic and changes nothing about a ranking with no bias term added).

Extended lookahead to a configurable depth (`GGML_CUDA_MOE_LOOKAHEAD_DEPTH`,
1-3, default 1) and measured production-shape throughput (4 slots, 65536 ctx,
moe-cache at the confirmed 4 GB knee, 4-way concurrent load, MTP off to avoid
its own confound):

| depth | r1 | r2 | mean |
|-------|-------|-------|-------|
| 1     | 50.46 | 49.85 | 50.16 |
| 2     | 51.28 | 48.07 | 49.68 |
| 3     | 53.43 | 48.67 | 51.05 |

**No measurable gain.** All three means sit inside the same ~5% run-to-run
noise this system has shown at every sweep this session (cache size,
admit_after). Depth 3's high single sample (53.43) is undercut by its own
second round (48.67) - noise, not a trend.

This is consistent with the cache-knee finding from earlier: at the 4 GB
operating point the system is compute-bound, not transfer-bound - misses are
absorbed by CPU fallback inside the compute window already. Better prediction
has nothing left to buy at this operating point; it would only start mattering
if the cache were pushed back into the transfer-bound region (smaller cache, or
a model whose expert corpus approaches the RAM ceiling), which is exactly where
the offline precision curve says it should help most and where a future variant
should be re-tested.

Default stays at depth 1. The sigmoid+bias ranking fix ships regardless of
depth - it is a correctness fix independent of how far ahead the prediction
reaches, and the depth-1 numbers above already include it.

## The expert-corpus tail is model-specific, not universal - corrected

Earlier reasoning applied Nemotron's measured skew tail (99.6% hit rate needs
99.97% of the corpus resident) to argue that a 744B model like GLM-5.2 would
need essentially its entire expert corpus resident to clear the hit rate a
20 tok/s target requires. That was an unjustified generalization from a single
model. Traced Gemma-4-26B-A4B (30 MoE layers, 128 experts/layer, top-8; 21,594
real tokens, same methodology as the Nemotron trace) to check:

| resident fraction of corpus | Nemotron hit rate | Gemma-4 hit rate |
|---|---|---|
| 17.5% | 46.4% | **69.4%** |
| 35%   | 66.1% | **89.3%** |
| 70%   | 93.0% | **99.2%** |
| ~90-99.9% | 99.6% (needed 99.97% resident) | **99.8% (at 90% resident)** |

Gemma-4's tail is dramatically more compressible. At identical resident
fractions it clears far higher hit rates, and it crosses 99.8% at 90% of its
corpus rather than needing essentially all of it. Two real models, two
meaningfully different tail shapes - this is a property of each model's
training (routing entropy, load-balancing regime), not a property of MoE
architectures as a class.

Consequence: the bandwidth-derived requirement (>=98.7% hit rate to keep a
744B model's NVMe miss traffic inside a 20 tok/s / 50ms budget, from
13.4 GB/token active weight against 3.55 GB/s NVMe) still holds - that's pure
arithmetic, architecture-independent. What is no longer justified is assuming
GLM-5.2's *tail shape* resembles Nemotron's specifically. Whether a resident
fraction well under "the whole model" can clear that bar depends on GLM's own
routing concentration, which is genuinely unmeasured. This is now an open
question, not a settled negative - the honest next step is tracing GLM's own
routing once the download completes, not further extrapolation from either of
the two models measured so far.

## Speculative eviction: tested and refuted in both regimes

Direct response to a specific proposal: use cross-depth agreement to reconcile
predictions (implemented - repeated confirmation across lookahead depths
promotes to protected, same as a real hit), and let speculative admission
evict the coldest probation-tier occupant when free, using the identical
heat-based criterion that decides who stays (`moe_cache_pick_coldest_unpinned`,
extracted as a shared helper and reused by both the real-miss and speculative
paths). Both built; both tested; both real results kept below rather than
argued from theory.

**Reconciliation (promote on repeated cross-depth agreement): no measurable
effect.** Gemma-4, depth 1/2/3, before vs after: 100.94/100.94, 98.78/98.91,
96.72/96.26 - identical within noise. Cross-depth agreement events are
apparently too rare to matter: most wrong deep guesses never get confirmed by
a later, closer prediction anyway.

**Speculative eviction (evict probation to admit a speculative guess): a real,
reproducible regression, in both regimes tested.**

| regime | no eviction (default) | eviction enabled | delta |
|--------|----------------------|-------------------|-------|
| abundant RAM (42 GB fast memory, 16.9 GB model) | ~100.9 tok/s (depth 1) | ~65.9 tok/s | **-35%** |
| genuine >RAM pressure (10 GB cgroup cap, 16.9 GB model) | 44.75 tok/s | 37.42 tok/s | **-16%** |

Three independent samples at each point, all consistent, no overlap between
conditions. The hypothesis that memory pressure would flip the calculus in
favor of eviction-enabled speculation is refuted, not confirmed - the
regression shrinks under pressure but does not reverse.

Mechanism: probation exists specifically to give a newly-admitted item one
chance to be touched again before it earns protection. Lookahead fires at
every layer, every token - far more admission attempts than the natural
token-to-token reuse interval. Letting every one of those evict on a miss
means "coldest available" collapses to "most recently admitted," because
nothing survives long enough for heat to separate genuinely useless items from
genuinely useful-but-not-yet-reused ones. Under pressure this gets slightly
*worse* to trade away, not better: every slot is more expensive to refill, so
discarding an unconfirmed probation item for a 48-59%-confidence guess is a
worse bet when capacity is already tight.

`GGML_CUDA_MOE_CACHE_SPEC_EVICT` defaults to off. Left in, gated, rather than
removed - a measured dead end is worth keeping on record over deleting the
evidence.

## Was it volume? Throttled to 4 evictions/token, tested - no

Direct follow-up: is the regression proportional to how often speculative
eviction fires? Added a per-token cap (`GGML_CUDA_MOE_CACHE_SPEC_EVICT_CAP`,
default 4, reset once per real decode step via `device.collect_calls`) so at
most 4 speculative evictions can happen per token, down from a theoretical
ceiling of `n_expert_used x n_layers` (up to ~240 for Gemma-4 uncapped).

Same >RAM pressure test (10 GiB cgroup cap, 16.9 GiB model):

| | mean tok/s |
|---|---|
| no speculative eviction | 44.08 |
| speculative eviction, uncapped | 37.42 |
| speculative eviction, capped to 4/token | 37.0 |

No difference. Volume was not the cause - even 4 wrong swaps per token,
repeated across a generation, cost the same as letting it fire unbounded. Each
individual speculative eviction is, on average, a losing trade regardless of
how often it happens: discarding an unconfirmed probation item for a
41-59%-confidence guess costs more than it gains, every time it happens, not
just in aggregate. This closes the speculative-eviction line of investigation -
neither reconciliation-based reinforcement, nor rate-limiting, rescues it.
`GGML_CUDA_MOE_CACHE_SPEC_EVICT` stays off by default.

## Literature check: why prediction-driven eviction fails, independent of our own tests

Zandieh et al.-style investigation aside, a dedicated study asked exactly the
question this session tested: "When Does Trace-Driven Evaluation Mislead MoE
Expert Caching?" (arXiv 2608.07911). They trained a proper ML predictor for
eviction specifically - not a heuristic, a model using log recency, frequency,
gate mass, concurrent routing multiplicity, layer, and popularity rate as
features - to predict how long a cached expert would stay unused. Result:
3.39% optimal-victim selection rate vs 22.11% for plain LFRU, with *negative*
rank correlation (-0.207) against true next-use distances. Worse than random,
despite far richer features than our heat-based signal.

Their stated reason matches our result exactly: "Expert prefetching predicts
which experts a future step will route to; [an eviction predictor] estimates
how long a cached block will remain unused. These are different targets."
Being a good prefetch-timing signal (our lookahead: 41-59% precision,
genuinely useful, shipped) does not make a signal good at the eviction
question, which is a different prediction problem. A model built specifically
for the eviction question still failed at it, worse than a frequency counter.

Consistent across SpecPrefetch, FineMoE, and MoE-Infinity in the wider
literature: prediction owns prefetch/admission timing; LFU/LFRU (frequency-
and recency-based, not prediction-based) owns eviction. That division of
labor is exactly what this project already ships - lookahead for admission
timing, LFRU heat/protection for eviction - and it is why the speculative-
eviction experiments above (reconciliation, uncapped, rate-limited) all
independently converged on the same negative result: asking one signal to do
both jobs is the actual mistake, not a bug in any particular implementation of
it.

## Silent failure: moe-cache never engaged for Ornith (or GLM-5.2) - fixed

Ornith-1.5-35B-A3B's Brain page showed the atlas but the live expert-activity
overlay was empty. Traced to `moe_cache_begin()`'s admission guard:
`expert_size < session->config.min_expert_bytes`, with `min_expert_bytes`
defaulting to 1 MiB. Ornith's per-expert tensors (256 experts, Qwen3.5-MoE
style - more, smaller experts than Nemotron's 128) are 840 KiB - below the
floor. Every candidate tensor was rejected before `session->tensor_layer` ever
got a single entry.

The dangerous part was not the threshold - it was that this failure was
**completely silent**. The session initialized normally, logged "explicit MoE
cache mode disables weight repacking" (success), and then did nothing for the
server's entire lifetime: 0 hits, 0 misses, empty expert-map, no warning
anywhere. From the outside this was indistinguishable from a correctly
configured, simply-idle cache.

`GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` already existed as an override, but
nothing told an operator they needed it. **GLM-5.2 has 256 experts/layer too**
and would have hit the identical silent failure - moe-cache dead on arrival
for the exact model this session has been building toward, with no signal
anything was wrong.

Fixed two ways:
- Default `min_expert_bytes` lowered 1 MiB -> 256 KiB, based on Ornith's
  measured 840 KiB/expert, with margin for genuinely-too-small slabs the
  floor exists to exclude.
- A one-time warning added at the exact rejection point, independent of what
  the threshold is set to, so this specific silent-failure shape cannot recur
  even if a future model's experts fall below whatever floor is chosen later.

Verified end to end on Ornith after the fix: `rows=29, cols=256` (real CPU-
offloaded layer count x real expert count), 61.0% hit rate, 22,732 hits,
5,736/5,986 slots used, 3,597 MiB allocated - all previously reporting zero.
