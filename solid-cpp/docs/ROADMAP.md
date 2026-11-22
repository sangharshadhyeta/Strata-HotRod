# solid.cpp roadmap and test ledger

Written 1 Oct 2026, updated 2 Oct 2026 01:xx. Numbers measured on this machine (RTX 3060 12 GB, i5-12400F 6C/12T, 30 GB RAM,
NVMe) with the paired method below unless marked "source" (another project's own claim, not measured here).

## Rules that govern the work

- Port hot-rod ONE step at a time; each step built, operator-tested and checked on gemma-4 first. Qwen is run on the final build.
- Never claim a gain that was not measured. A commit message states what was verified and what was not.
- Never weaken a test to make it pass. Never trust a single run: gemma greedy output is not bit-reproducible (same setting twice
  agrees 21-67% depending on the run - always carry a same-build control). Qwen is even noisier at small N (n=5 agreed
  38-55% across settings with no clean separation) - treat small-N Qwen fidelity numbers as unreliable; its large-N aggregate
  counters (cache hit rate, summed over hundreds of thousands of hits/misses) are reliable even when fidelity isn't.
- Take from other projects only what is better than ours; keep our design where it is better.
- A failed build must stop a test chain. A structural plumbing change to a hot dispatch path (changing a callback signature
  used by every call site) does not get rushed in at the end of a session - scope it, document it, do it fresh.
- Kill exact PIDs, never `pkill -f` (it can match your own wrapper's command line and self-abort). One background task per
  wait. Before trusting ANY benchmark number, check nothing else is contending for the same disk/GPU - a duplicated,
  forgotten background download on the same physical NVMe as the model files under test silently degraded an entire Qwen
  run's timing numbers until caught (2 Oct). ThinkVest stays stopped while the model is being worked on.
- Large downloads for comparison files go on a disk OTHER than the one holding the models under active test (here: `/home`,
  not `/mnt/nvme` - verified via `lsblk` to be a genuinely separate physical device, not just a different mountpoint on the
  same disk).

## Method

Paired A/B: same prompt and seed per request across settings, an identical-config control (`*_again` or `exact_ref_again`),
warm median plus cold (first two requests), mean +-se and requests-faster counts. Fidelity checks: an EXACT unbiased
reference (every lossy feature off), greedy decode, prefix-agreement against it, always read against that setting's own
noise-floor control before concluding anything. Harness: `ab (the local A/B harness, not included)` (ab2.py, tally.py,
stepcheck.py, longctx.py, gemma256.py, fidelity.py, biascheck.py, biascheck_qwen.py). Chat endpoint with the chat template.

## Wins in the tree (gemma-4, 64K context, 2 slots, unless noted)

| Change | Measured |
|---|---|
| Shipped defaults overall | 77.6/78.2/77.1 tok/s (code+reasoning, was 54.8/56.2); 59.3/62.5/59.9 (mixed, was 29.0/33.5/31.3) |
| Neuron subsetting off unless calibrated on | cost ~33%; ~+50% when off |
| Draft depth 4 (was 8) | depth 8 is -11% +-2 on the shipped stack; +22-32% earlier |
| Draft confidence gate p_min 0.5 | +13% |
| Do not pin `--spec-draft-n-max 8` in ThinkVest's launch | +15% prose, 0% code |
| `-md` naming an MTP head runs as one (header-based, not filename) | -25% without it |
| MoE model that fits 3/5 of total RAM is read into memory, not mapped | -13% without it |
| Expert-cache VRAM reserve floor 1,500 MiB | two ~32K-token prompts complete; no speed cost |
| Draft VRAM reserved in the serving placement probes | launch without `-ncmoe` no longer crashes |
| 64K minimum context | fit floor and fallbacks were 4,096 |
| PART_STATS (read-only counter) | gemma: 5.1% partial expert residency, 8.6% rows miss, 69 MiB/token fetched |
| mmid/mmf race fix | MUL_MAT_ID 869/869, MUL_MAT 1186/1186 pass; +2.8% vs 0.3% control (tie, no regression) |
| Lazy-KV commit-failure yield (`ggml_backend_cuda_moe_cache_yield_vram`) | 2 concurrent ~30K-token prompts: server lives with it, dies without it (confirmed log: pool released, commit retried) |
| Lazy KV default ON for windows >=131072 cells, OFF below | -c 262144 short prompts: +15.1% +-1.6, 1315 vs 835 slots, hit 0.66 vs 0.53. -c 65536: tie (stays off). **3-way verified 2 Oct**: no-env-var auto run (975 slots/hit 0.586/49.8 tok/s) matches forced-ON (986/0.578/52.8) and is clearly unlike forced-OFF (511/0.349/39.5) - auto-detection confirmed correct |
| 9-commit upstream group (loader RAM-peak fix, GGUF load speedup, RMS_NORM+SCALE fusion, MoE-reduction fusion, fusion extended to spec-decode batches, row prefetch, 0-sized-ids skip, draft-context-not-embedding fix) + SWIGLU_CLAMP adaptation | builds clean, all 6 touched operators 100% pass, +3.0% +-1.2 vs 4.9% control (tie - no regression, no proven gain on gemma; this group's payoff is mostly Qwen-side: RAM-peak fix, GGUF load speed, row prefetch) |
| Router bias toward cache-resident experts (`LLAMA_MOE_CACHE_BIAS=beta`, decode-only) | **See "Router bias vs substitution" below - the full story, corrected after an initial test ran prefill-biased by mistake.** |

## Measured, not wins (kept, off or neutral)

Pre-router (tie gemma; Qwen -7% but unverified whether the saved weights even loaded - RECHECK), neuron subsetting (-33%
gemma, -5% Qwen), `-ub 1024` (+5% decode but -22% per request on 5K-token prompts), drafter n-gram cascade, pinned cores,
5 threads, prediction ring, shallow-layer eviction bonus (Fate-style: tie at 2x, -2.4% at 4x on gemma - removed from tree),
adaptive draft-length governor (MoE-SpeQ-style bandit over depths 2-8: **-4.2% +-1.4 vs fixed depth 4, 4 of 16 faster** -
it explores depths that lose and depth 4 is already the best arm on gemma's mix; kept in tree, OFF by default).

## Router bias vs substitution - the real comparison

Substitution's documented +8.11% win (`DEFAULT ON` comment in moe-cache.cu) was measured on **Ornith-1.5-35B-A3B**
(256 fine-grained experts, qwen35moe arch) - never on gemma-4 or Qwen3.8-Flash-Next. That model has far more expert
redundancy (a stand-in is a close approximation) and a much higher natural miss rate than gemma's 0.84-0.86 hit rate.
Substitution was never actually validated in gemma's or Qwen's regime; today's results don't contradict the original
finding, they just show it doesn't generalize to these two models as shipped.

First bias test accidentally biased PREFILL too (not just decode) - corrected (`391f116f0`): bias now only touches
decode-sized batches (`LLAMA_MOE_CACHE_BIAS_MAX_BATCH`, default 8 tokens), matching the paper and the lookahead gate's
own convention.

**Gemma-4** (exact reference = substitution off entirely, n=10 prompts, greedy):

| Setting | Agreement vs exact ref | Speed vs ref | Hit rate |
|---|---|---|---|
| Exact ref, run again (noise floor) | 67.1% | +0% | 0.847 |
| Substitution (shipped default) | 21.0% | +1.6% | 0.848 |
| Bias 0.5, decode-only, subst off | **31.1%** | +2.6% | 0.894 |
| Bias 1.0, decode-only, subst off | 27.8% | +3.9% | 0.917 |
| Bias 1.0 + substitution also on | 20.3% | +1.5% | 0.916 |

**Qwen3.8-Flash-Next** (n=5 prompts - CAUTION, see contamination note below):

| Setting | Agreement vs ref | Speed vs ref | Hit rate (large-N, reliable) |
|---|---|---|---|
| Exact ref | 100.0% | +0% | 0.262 |
| Exact ref again (noise floor) | 41.5% | +2.3% | 0.256 |
| Substitution (shipped) | 54.6% | -1.0% | 0.254 |
| Bias 0.5, subst off | 38.6% | +1.5% | **0.330** |
| Bias 0.5 + substitution fallback | 45.6% | +6.2% | **0.331** |

**What's solid on both models:** substitution never moves the cache hit rate (by construction - it answers a miss
differently, it doesn't prevent one). Bias DOES move it (gemma 0.847->0.917 at beta=1.0; Qwen 0.26->0.33 at beta=0.5) -
a real, large-N-backed, mechanism-consistent result on both models. Stacking bias+substitution is never better than bias
alone (gemma: 20.3% < 27.8%; Qwen: hit rate identical, 0.331 vs 0.330 - exactly what "can't move hit rate by construction"
predicts for the fallback case too).

**What's NOT solid yet:** the Qwen fidelity/speed table above was run while two duplicate background download processes
(an earlier bug - see Rules) were hammering the SAME NVMe drive Qwen's disk-backed experts stream from. Found and fixed
mid-run (2 Oct ~01:00): killed the stale duplicate, moved the download entirely off `/mnt/nvme` to `/home` (separate
physical disk, verified via lsblk). The Qwen speed/fidelity numbers above should be treated as contaminated by I/O
contention (noise-floor control itself swung +2.3% with zero real difference) and **rerun clean before trusting them** -
the hit-rate numbers are fine (large-N aggregate, not sensitive to per-request timing jitter).

**Decision: NOT made a default yet.** Bias beats substitution on every measure that survived testing on both models. It is
a legitimate candidate to replace substitution as the default miss-handling strategy, pending: (1) a clean Qwen
speed/fidelity rerun now that the disk is quiet, (2) an owner decision, since this changes default output on every request.
Both stay opt-in flags; substitution is NOT deleted (kept as a documented fallback option either way this resolves).

## L2 (HOBBIT cumulative-gate-mass skip) - approximation built and tested, no gain

The full version (live per-token gate weight reaching the planner) needs a `GGML_OP_MUL_MAT_ID` arity change - every
backend, every MoE model builds on that op's fixed 3-source signature - declined as disproportionate risk. Built a
safer static-profile approximation instead: a per-rank average gate-mass table measured on gemma-4 via the existing
`llama-debug` eval-callback tool (1,620 real token x layer samples), gating substitution eligibility by cumulative
mass instead of raw rank when `GGML_CUDA_MOE_CACHE_SUBSTITUTE_MASS_THRESHOLD` is set (-1 = off, default). Paired test
vs the existing rank floor on gemma-4: ties or slight losses at mass thresholds 0.60/0.77; the apparent gain at 0.93
(+2.6%) traced to run-to-run VRAM-budget variance (confirmed via `dispatch_failures` and allocated/budget log lines),
not the gating mechanism. No clear gain over the rank floor; kept in tree, off by default, same pattern as the
governor and shallow-layer bonus.

## Atlas

Lookahead and eviction protection ON by default; warming (cold-start only) and the admission ring OFF. Not re-measured
this session.

## Hot-rod port status

Merged to main: step 1 (argsort + flash-attention barrier), 2a (sparse FA path), 2b (sparse FA, qwen4exp head 256),
PART_STATS (+ format-string repair), mmid/mmf race fix, the 9-commit upstream group (see table above), the L2
mass-threshold approximation (tested, no clear gain, kept off by default - see below), and **the full KV streaming
chain S3a-S3e** (`--kv-stream`/`--kv-resident`, qwen4exp-only; gemma-checked at every step for build/ops/clean-refusal,
all off-by-default no-ops confirmed). Two real bugs were found and fixed while porting S3e, not worked around: a
register-pressure regression in the F16 flash-attention kernel from merging in the q8_0 dequant path (fixed by
splitting it into its own function), and a shared-memory opt-in gated behind one flag shared by two different kernel
pointers, so the q8_0 gather kernel's own `cudaFuncSetAttribute` call was silently skipped whenever an F16 call ran
first - not a hardware limit, a missed opt-in. `port/hotrod-kv` and `port/hotrod-upstream` are now fully merged;
worktrees pending cleanup (see task list).

**Separately found and fixed while running the Qwen final validation (2 Oct, this session):** `ggml_moe_cache_trim`
(the OOM-fallback the generic CUDA pool allocators call) marked the ENTIRE expert cache device permanently dead on the
first narrowly-failed allocation, instead of freeing just enough - a realistic event during Qwen's tight model load on
a 12 GB card. This silently zeroed the cache (`budget=0`, hit_rate=0.0, ~3 tok/s) for the rest of the process. Fixed:
the two call sites in `ggml-cuda.cu` (legacy pool allocator, VMM pool allocator) now use the existing non-destructive
`ggml_backend_cuda_moe_cache_yield_vram` instead, which only frees as much as the specific allocation needs. Confirmed
fix restores cache function (pools populate, hit_rate > 0) but the resulting budget (~1-1.4 GiB) is still well below
the hit rates (0.26-0.33) measured in earlier sessions at the same context sizes - tested directly, not explained by
context length (16384 vs 65536 both land in the same 1-1.4 GiB range) or by the new KV-streaming code (confirmed
properly gated off). Consistent with the known run-to-run VRAM-snapshot variance already documented elsewhere in this
file, not a further bug found via manual search - see the Integration survey below for where the deeper structural gap
likely is.

## Item 4 (concurrent expert reads) - PART_STATS gate now satisfied, scoped not attempted this session

ROADMAP's own "Next, in order" list gated this behind PART_STATS confirming the SSD is actually the bottleneck. That
confirmation now exists, measured directly (2 Oct): a live Qwen run (6-turn, ~12,700-token conversation) read 617.6 GB
total from the model's 45 GB weight file - roughly 14x reread, not a bug but a direct consequence of the expert cache
(1,350 slots) being far smaller than the model's full expert population across 48 layers, so experts get evicted
(39,172 evictions this run) and refetched repeatedly as the working set rotates. That reread traffic averaged ~0.56
GB/s over the run's wall-clock span, against a measured 2.0 GB/s raw sequential `dd ... iflag=direct` read on the
same NVMe - roughly 28% of the drive's demonstrated capability. The gate is satisfied: the SSD has real headroom this
fork isn't using.

Root cause, confirmed by reading the actual fetch path (not assumed): `moe_cache_dispatch`'s cold fills run on exactly
one fill-worker thread per device (`moe_cache_device::worker`, a single `std::thread`), and by default experts are
read via mmap demand-paging - the H2D copy's own memory access faults in 4 KiB pages one at a time, with no readahead
sized to the actual expert (687-900 KiB per the pools measured this session). A single thread, page-faulting in 4 KiB
increments, cannot generate the queue depth an NVMe needs to approach its rated throughput - this is sufficient on its
own to explain the measured 28% utilization, independent of the cache-size question above.

Two levers exist, already distinguishable by risk:

1. **Already built, off by default, not yet tested on this card**: `GGML_CUDA_MOE_CACHE_EXPLICIT_READ=1` (code at
   `moe-cache.cu`, search `moe_cache_explicit_read_impl`) resolves the mmap's backing file via `/proc/self/maps` and
   does one `pread()` sized to the whole expert into the pinned staging buffer, instead of letting the copy fault
   pages in one at a time - the exact TurboFieldfare-measured gap (0.5 vs 4-6 tok/s on a from-scratch Metal/Swift
   MoE-on-SSD runtime, unrelated project, same underlying problem) this was written to close. Still single-threaded
   (one pread() at a time on the one fill worker), so it addresses the "4 KiB at a time, no readahead" half of the
   problem but not the "no queue depth" half - a real, low-risk, already-safe (comment confirms "no correctness
   implications beyond did the read succeed") thing to measure before building anything new.
2. **Not built**: real queue depth needs more than one read in flight at once. The lowest-risk design that does not
   touch the single fill worker's cache bookkeeping/locking/eviction logic at all: a small pool of dedicated I/O
   threads (2-4) that only `pread()` queued jobs' bytes ahead into pinned staging buffers, so the fill worker's own
   H2D copy never blocks on disk - it copies already-staged bytes. This keeps the existing single-threaded
   correctness model (one thread owns the cache's mutable state) while letting the kernel/NVMe see several
   outstanding reads at once, which is what queue depth actually means. A more invasive alternative (io_uring/Linux
   AIO for true single-thread-issued async reads) would avoid the thread-pool's synchronization entirely but is a
   much larger, more portability-sensitive change for a first attempt - note it as the fallback if the thread-pool
   approach doesn't close enough of the gap, not the starting design.

**Not attempted this session** - this is a structural change to the hot fill-worker path, the same area that already
produced two real bugs today (S3e's register-pressure regression, the shared-memory-flag bug) from being touched
under time pressure. Per this file's own rule ("a structural plumbing change to a hot dispatch path... does not get
rushed in at the end of a session"), the next session should: (a) test lever 1 (`EXPLICIT_READ=1`) first, cheaply,
since it already exists; (b) if real gap remains, build lever 2 on a worktree, prove it against `test-backend-ops`
and a paired gemma/Qwen comparison before touching main, same process as the KV-streaming port.

## Integration survey (2 Oct) - SURVEY ONLY, not yet verified or acted on

Prompted by the live VRAM-budget bug above: the system has grown many independently-tuned subsystems (expert cache,
KV streaming, speculative draft depth, three separate expert-prediction signals, six independent slot-trust verifiers)
that don't share state with each other, each reasoning about its own slice of the same fixed hardware budget in
isolation. Three background surveys (code-reading only, no changes) were run across `moe-cache.cu`'s ~14.7k lines and
119 tunables, `llama-kv-cache.cpp`, `ggml-cuda.cu`, `llama-graph.cpp`, and `speculative.cpp`, looking for (a) more
instances of the uncoordinated-resource-claim bug just fixed, (b) prediction/prefetch signals that could inform each
other but don't, and (c) default-on mechanisms the project's own data already shows losing.

**IMPORTANT: these are agent-read findings, not independently re-verified.** Two were already caught self-contradicting
or stale during the survey itself (a claimed-removed mechanism that's actually still in the tree off-by-default; a
"new" finding that was already in this file). Before acting on ANY item below: re-read the cited code directly (line
numbers drift, agents can misread a guard), and where a performance/quality claim is involved, run a real paired test
against the noise floor per the Method above - do not trust a prior number or an agent's restated number without
re-measuring.

**Resource arbitration (VRAM + pinned host RAM) - highest confidence, most severe:**
- Pinned host RAM has two fully independent, uncoordinated claimants: the KV-streaming buffer (`ggml-cuda.cu`,
  `ggml_backend_cuda_kv_host_buffer_type_alloc_buffer`, sized from context length at KV-construction time, no live
  memory check at all) and moe-cache's own `GGML_CUDA_MOE_CACHE_HOST_MB`/`PIN_MB` (the latter does read live
  `/proc/meminfo`, but only once, lazily, and only runs after KV-streaming has already taken its share). Pinned pages
  are unreclaimable by the kernel - over-commit here risks the OOM killer ending the whole process, not a graceful
  degradation. Worse than the VRAM bug just fixed.
- `cudaGraphInstantiate`/`cudaGraphLaunch` (`ggml-cuda.cu`) are wrapped in `CUDA_CHECK`, which aborts the process on
  any CUDA error including OOM - with no `yield_vram` call anywhere nearby. Protected only by the budget calc's
  guessed static safety margin, which the margin's own code comment documents having been wrong once already in
  production (1 Oct, a server left with 381 MiB free died in `cudaGraphInstantiate`).
- `moe_cache_grow_device` (dispatch scratch growth) `cudaMalloc`s directly without consulting `device.budget_limit` -
  self-documented in the code's own comment as "growth escapes the budget." `GGML_CUDA_MOE_CACHE_PRESIZE_X` can turn
  this into one large, budget-blind allocation on the decode thread - a plausible trigger for the OOM path. Failure
  mode is non-fatal (degrades one dispatch) but doesn't cooperate with the arbiter it just stressed.
- The generic weight-buffer allocator (model/draft weights, non-lazy KV) has no yield_vram fallback - a clean load
  failure, not silent corruption, but still a real gap against the budget authority.
- Neuron-reduce was checked and is a false positive: it correctly draws from `device.budget_limit`, same authority as
  everything else - not an independent consumer.

**Prediction/prefetch signals - mixed, one clear highest-leverage item:**
- **Static admission/eviction weights never get fed back from already-collected accuracy data.** The per-signal hit
  counters (`substitute_rank_hits`, `substitute_atlas_hits`, `predictor_top1_hits`, `lookahead_hits`) update every
  step and are already computed - they're just printed via `GGML_CUDA_MOE_CACHE_SUMMARY` and discarded, never fed back
  into the static `SUB_W_ATLAS`/`SUB_W_COACT`/`PREDICTOR_*_WEIGHT` constants. Cheapest fix of everything surveyed: the
  measurement exists, only the feedback wiring is missing.
- No unified per-slot trust state: six independent verifiers (`WEIGHT_GUARD`, `SLOT_NAN_SWEEP`, `CANARY`, `GEN_CHECK`,
  `MAP_AUDIT`, `PIN_AUDIT`) each just `fprintf` and increment their own counter on failure; none write into shared
  slot state that substitution/eviction could consult. Real gap, but low current cost (debug-tier tooling for an
  already-fixed corruption bug, not live-path logic).
- Four prefetch paths (`READAHEAD`, `CPU_PREFETCH`, `LOOKAHEAD_DISK_PREFETCH`, `LIVE_PREFETCH`) have no shared
  in-flight ledger, but this is mostly a false alarm: all four only issue `posix_madvise(WILLNEED)`, an idempotent
  kernel-deduped hint, not a real transfer - actual VRAM copies are correctly gated per-mechanism. The one live risk
  is concurrent-prefetcher page-cache contention (READAHEAD alone already measured an 8% warm-throughput regression),
  not duplicate bandwidth.
- Draft-length governor vs cache pressure: confirmed real (the governor never reads `moe_cache_*` state), but moot -
  the governor already lost on its own economics (-4.2%) independent of this gap, and **is still in the tree,
  off by default** (NOT removed - an earlier note in this file's history claiming removal was wrong; corrected here).
- Router bias / substitution / prerouter: mostly a false positive on first read - `docs/ROADMAP.md` above already
  shows bias and substitution were tested stacked together, not only independently, and the prerouter already has one
  narrow admission hook (`PREDICTOR_SUB_WEIGHT`, currently 0.0/disabled). The one real residual gap: router bias reads
  aggregate resident-expert counts but has no idea *why* an expert is resident (lookahead vs prerouter vs atlas vs
  plain reuse) - minor, deliberate-design-adjacent.
- "Spec-agreement eviction" vs the draft-length governor: a naming collision, not a coordination gap -
  `SPEC_AGREE_CONF`/`SPEC_EVICT*` concern expert-routing lookahead agreement (do two prediction signals pick the same
  expert), unrelated to LLM token-level speculative decoding. Zero shared code either way.

**Default-on mechanisms the project's own data already shows losing - needs an owner decision, not more surveying:**
- **Substitution ships on by default** despite the "Router bias vs substitution" table above showing it at 21.0%
  greedy agreement (vs a 67.1% same-build noise floor) for only +1.6% speed on gemma, and -1.0% speed on Qwen - the
  project's own already-measured data, not a new agent claim. The `+8.11%` win cited in its `DEFAULT ON` code comment
  was measured on a different, now-retired model (Ornith-1.5-35B) and never validated on either currently-tracked
  model. This file's own "Next, in order" list (below) has called this an open decision since 1 Oct; the survey adds
  no new data, just flags that the decision is still pending and the case for flipping it is already strong.
  Substitution's pace-EMA bookkeeping (`moe_cache_track_fetch_pace`, called unconditionally from
  `ggml-backend.cpp` on every H2D expert copy) is paid even when `SUBSTITUTE=0` - small today, but dead weight if
  substitution is ever defaulted off.
- **Prerouter eviction-weight default (1.0, i.e. fully on)** rests on a result ("WON at eviction protection, 71.41
  tok/s") that this session's own re-test under a proper paired/noise-floor-controlled methodology could not
  reproduce (tie on gemma, -7% on Qwen) - and per the agent survey, this RECHECK is **already flagged in this file**
  (see "Measured, not wins" above), not a new finding. Still unresolved either way - needs the recheck actually run,
  not another survey.
- Atlas-warm-burst (9 tunables, `ATLAS_WARM_BURST_*`): correctly off by default already, costs nothing today. Its
  evidence is two single-shot ablations from 29 Aug, predating this project's own paired-A/B method adopted 1 Oct.
  Lower urgency than the above two (no default-path cost), but a candidate for either a real re-test on the two
  current models or deletion as unvalidated surface area.

**What this is NOT yet:** a plan, or a decision. It's a prioritized list of what to re-verify first. The highest
dollar-per-hour items are: (1) flip substitution off by default - the data to support it already exists in this file;
(2) actually run the prerouter eviction-weight RECHECK that's been pending since before this survey; (3) wire the
already-collected per-signal accuracy counters into the static admission weights; (4) scope a shared VRAM+host-RAM
ledger that the KV-stream buffer, moe-cache's host/pin budgets, and the generic CUDA allocators all register against,
closing the pinned-RAM and graph-instantiate gaps at the same time instead of patching each OOM site individually.

## Next, in order

1. Re-verify the Integration survey's highest-leverage items (above) with fresh code reads and real paired tests
   before acting on any of them - do not trust the restated numbers.
2. Clean Qwen speed/fidelity rerun (bias vs substitution) now that the duplicate-download disk contention is fixed.
3. Owner decision: flip substitution off by default (data already supports it) vs keep as-is.
4. Actually run the prerouter eviction-weight RECHECK (pending since before 2 Oct, confirmed still open by the survey).
5. Scope a shared VRAM+host-RAM resource ledger (the highest-leverage structural fix from the survey) as a proper
   structural change, not rushed in - per the rules above, scope it, document it, build it fresh on a worktree.
6. Concurrent O_DIRECT expert reads + sidecar per-expert pack (SSD-LLaMA/Strata idea) - Qwen-only, gated on PART_STATS
   confirming the SSD is actually the bottleneck there.
7. Keep `docs/index.html` to tested numbers only (`docs_audit.py` regenerates it); remove the `port/hotrod-kv` and
   `tmp/hr`/`tmp/hr2` worktrees now that the full chain is merged.

## Environment facts

- The Qwen GSQ-RCO IQ3_S comparison file (ISTA-DASLab, 83.6 GB) is downloading on `/home` (NOT `/mnt/nvme` - that disk
  hosts the models under active test). Network ~1 MB/s observed; resumable (`curl -C -`), sha256-checked on completion.
- Gemma 256K, one slot, dense KV: completes (240K tokens, prefill 422s / 657 tok/s), but only 510 expert-cache slots
  (hit 0.31) and 12.8-15.8 tok/s - the KV starves the cache at full-window usage. The large-window default above targets
  the SHORT-prompt-at-large-window case, not this one; a full 240K prompt is still the one case lazy KV doesn't help.
- ThinkVest launch (`ThinkVest/scripts/services.sh`) no longer pins depth 8 / `--spec-prob-accept` (committed there).
