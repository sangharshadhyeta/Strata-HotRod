# Plan: work stemming from the VRAM-item table

Context: a table auditing what lives in VRAM (ideal case: model fits entirely;
our case: larger-than-VRAM MoE model, this fork's actual placement/cache
stack) surfaced several concrete proposals for the "our case" column. This
file tracks the resulting build plan, one track per proposal, so it survives
across sessions instead of living only in chat history.

## Track 1 — Atlas-driven cache warming

Stems from the "MoE expert weights" row: use the already-existing Expert
Atlas (per-expert topic-affinity data, measured offline by
`llama-expert-atlas`, feeding the Brain/Atlas UI) as an extra promotion
signal for the VRAM expert cache, on top of plain LFRU heat.

1. **[done]** Step 0 — live request-direction tracking. `ggml_moe_cache.set_atlas`
   registers each expert's measured `(x, y, spec)`; moe-cache maintains a
   live, decaying centroid (`req_dir_x/y`) of where the current request is
   trending in that space, updated on every real routing decision (hit or
   miss). Purely observational — nothing reads it yet. Verified end to end
   on a live server (60 tensors registered, centroid genuinely tracks real
   traffic across a 200-token generation, no crash). Commit `5fc3f803e`.
2. **[done]** Expose `req_dir_x/y` via `/experts` (`stats.req_dir`, omitted
   until at least one atlas-covered expert has been selected). Verified live:
   absent before generation, real coordinates after. Commit `5b739f7dc`.
   Frontend rendering (an actual marker drawn on the canvas) is not part of
   this - only the data is available so far.
3. **[done]** Warming action — `moe_cache_atlas_warm`, rate-limited to once
   every 64 `plan()` calls, ranks a tensor's atlas-covered candidates by
   cosine similarity to `req_dir` (discounted by specialization confidence),
   admits top-K into free slots only - **never evicts**. Off by default
   (`GGML_CUDA_MOE_CACHE_ATLAS_WARM=1` to enable), not yet A/B'd for a real
   effect, same discipline as everything else predictive here. Caught and
   fixed a real bug during first live test: atlas expert indices weren't
   bounds-checked against the tensor's actual expert count, causing an
   out-of-bounds read (confirmed segfault, not hypothetical) when tested
   with a mismatched model/atlas pair - fixed with the same `expert <
   n_expert` guard every other caller already had. Verified clean afterward
   across two real generations (800 tokens total), no crash, hit rate
   climbing to 94%.
4. **[done]** Pairwise co-activation ("fire together")
   signal. Split in two once actually scoped:
   - **4a, done**: within-layer co-activation - which experts get selected
     *together* in the same real routing decision (`device.co_activation`,
     an unordered-pair count keyed by tensor + both expert indices). Capped
     to `n_ids <= 32` so a large prefill batch's O(n²) pass over its own ids
     never runs - verified no slowdown on a ~2000-token prefill (1.8s, no
     crash) and no crash across normal decode traffic either. Observational
     only, same as req_dir - nothing reads it yet.
   - **4b, done**: the token/batch-boundary prerequisite turned out to
     already exist - `moe_cache_session_enter` is called exactly once per
     real `ggml_backend_sched_compute_splits` invocation (one graph compute
     = one forward pass/batch) via an existing RAII scope in
     `ggml-backend.cpp`, so resetting `last_top_expert_valid` there needed
     no new API surface at all. Cross-layer edges tracked using `ids[0]`
     (rank-ordered router output, index 0 = top pick) as each layer's
     representative expert, into `device.co_activation_cross_layer`.
     Imperfect for a multi-sequence batched compute call (mixes different
     sequences into one shared "last layer" state for that call) - same
     granularity limit `req_dir_x/y` already has as a single per-device
     value, not pretending otherwise. Verified live: 3 concurrent requests
     (the case most likely to expose issues in session-wide state), a real
     ~2000-token prefill, no crash in either. This is the more direct answer
     to "why can't the atlas be dynamic" than nudging static positions from
     a live-classified topic would be.
5. **[built, then REVERTED]** 3D topic-space visualization (`ec1354bc1`,
   reverted in `de08f87b2`). Built as a real 3D sphere with topic (not layer)
   on the third axis, Fibonacci-sphere anchors, orbit/zoom. Reverted at the
   user's request: too taxing on a machine that is usually also serving the
   model. The cost I had underweighted - the full atlas ships in the
   `/experts` response on **every 1.5s poll**, so adding the per-category
   `cats{}` vector inflated that payload from 1.2MB to 2.0MB per poll per
   open tab, on top of the per-frame draw cost. The active atlas file was
   reverted to the pre-`cats` v1 for the same reason.
   Kept (data-side, no per-frame or per-poll cost): the generator still
   emits `cats{}` (opt-in by regenerating), the co-activation export API,
   and the server-side tensor->layer translation. Those are what Track 1.6
   would build on. If revisited, WebGL would suit an interactive connectome
   better than the 2D-canvas projection used here.

**Same-layer artifact in cross-layer edges — FIXED** (`9f68d2f64`). One
logical layer dispatches several expert tensors (gate_up_exps then
down_exps) off a single router decision, so consecutive `plan()` calls were
frequently the same layer and got recorded as self-edges - measured at 55 of
64 exported edges on Ornith and 45 of 64 on Gemma, i.e. ~86% of the signal
was artifact. Fixed by parsing the logical layer from the tensor name
`begin()` already receives (llama.cpp names every per-layer tensor
`blk.<layer>.<what>.weight`, so no new API parameter was needed) and only
recording an edge when the layer actually changes; an unparseable name
yields -1 and records nothing rather than guessing. Verified live on
Ornith: **64 of 64 exported cross-layer edges are now genuine transitions,
0 self-edges** (was 9 of 64 genuine) - a 7x increase in usable signal.

**Step 5 prerequisites (done)**: storage bug fixed (`moe_cache_edge`
replaces the original opaque-hash counters for both `co_activation` and
`co_activation_cross_layer`); export API added
(`ggml_moe_cache.get_co_activation`, top-K by count, real tensor identity);
server-side tensor->layer translation done (`atlas_tensor_layer` map,
reusing the mapping already built for `set_atlas` registration).

**Step 3's real-world verdict, tested twice, two different data sources**:
- *Mismatched data* (Ornith's atlas on Gemma - arbitrary topic labels, no
  real relationship to what Gemma's experts do): flat-to-noise on both
  tok/s and hit rate, both the conservative and looser "speculative" eviction
  variants, across 2 interleaved rounds. See the git log for the full numbers.
- *Matched data* (Gemma's own atlas, `/mnt/nvme/models/moe-test/expert-atlas.json`,
  n_layer=30, on Gemma): a materially different picture. The conservative
  variant (`GGML_CUDA_MOE_CACHE_ATLAS_WARM=1`, defaults `EVICT_MIN=0.6
  EVICT_CAP=1`) beat baseline tok/s in **both** interleaved rounds (58.44 vs
  58.28, 58.49 vs 58.33 - consistently, not a coin flip like the mismatched
  test), a small but real +0.29% average. The looser "speculative" variant
  (`EVICT_MIN=0.0 EVICT_CAP=2`) won hit rate consistently (~93% vs ~91%
  baseline in both rounds) but showed real instability on tok/s - a 54.8
  tok/s outlier in round 1 (a genuine ~6% drop, not noise), suggesting more
  frequent eviction sometimes costs more than the hit-rate gain returns.
  Only 2 rounds so far - real, consistent direction, not yet enough samples
  to call fully proven. Needs a longer run (more rounds, larger/richer atlas)
  before treating this as validated - in progress.
- **Update, 4-round re-test on the same matched data**: the 2-round signal
  did not hold up. Conservative: tok/s 58.32 avg, exactly tied with
  baseline's 58.32 avg, full range overlap (58.07-58.50 vs 57.97-58.53).
  Speculative (n=3, one round lost to an unrelated build-vs-test race, not
  the mechanism's fault): hit rate 93.47%, now *below* baseline's 94.86%,
  reversed from the earlier apparent win. Textbook regression-to-the-mean -
  a small early sample looked like a trend by chance. **Corrected verdict:
  no measurable benefit from atlas warming on real matched data, at this
  sample size. Not proven harmful either - everything sits within noise of
  baseline.** Stays off by default. Documented plainly rather than kept as
  the earlier, more optimistic read.
- **Final, n=8**: baseline 58.10 tok/s / 94.83% hit rate, conservative 58.29
  / 94.19%, speculative 58.00 / 94.05%, spec_agree 58.34 / 94.30%. All four
  configs land within a ~0.34 tok/s / ~0.78pp band - confirms flat. Tested
  on a steady single-topic workload, where LFRU alone is already near-
  optimal - see the design-flaw note below for why this doesn't settle
  whether the *corrected* design (below) would do better on a topic-switch
  workload, which is what it's actually meant for.

**Real design flaw found, not yet fixed**: `moe_cache_atlas_warm` competes
for free slots and evicts probation candidates to admit *new* experts - the
same operation router-lookahead prefetch already does, just with a weaker
prediction signal (cosine similarity to a decaying centroid vs. the real
router's own one-layer-ahead logits, 59.3% precision at depth 1). That's
structural redundancy with an existing, better mechanism, not a distinct
purpose - explains why the A/B tests kept coming back flat regardless of
tuning. The right purpose for topic-affinity data is different: not "what to
fetch" (already router-lookahead's job) but **"what to keep"** - of experts
already resident and cooling toward eviction, protect the ones topically
aligned with where the request is heading, even if their raw LFRU heat says
they're not urgent. That's a genuinely different signal neither LFRU nor
router-lookahead has access to. The correct integration point is
`moe_cache_weighted_heat`/`moe_cache_cost_tier_weight` (already built earlier
today for the NVMe-vs-RAM eviction cost tier) - topic alignment should be
another factor in eviction-candidate scoring there, not a second competing
admission path. Real redesign needed, not a tuning fix - not started.

Separately, all A/B testing so far used a single fixed-topic prompt
generated start to finish - exactly the regime where LFRU's reactive signal
is already near-optimal (nothing changes, nothing to predict) and where
atlas warming (in either the current or the corrected design) has no room to
show a difference. Built a topic-switch test (code-heavy prompt, then an
immediate hard pivot to medicine, measuring the post-switch request
specifically) and ran it against the OLD (pre-redesign, admission-competing)
mechanism before removing it from source. **Real result, not noise**: hit
rate right after the switch was higher with warming on in **4 of 4 rounds**,
no exceptions (89.31% avg vs 86.56% off, +2.75pp), at a small tok/s cost
(58.22 vs 58.40, -0.3%). This is a genuinely different, much stronger signal
than anything the steady-state tests showed - confirms the workload
diagnosis was right, and that the underlying concept has real value even in
the flawed (redundant-with-prefetch) design. Re-testing the corrected
eviction-weighting design against this same topic-switch workload next, to
see whether it preserves or improves this result.

**Resolution - three-way comparison, same topic-switch workload, same
methodology throughout:**

| Variant | Rounds won (of 4) | Hit-rate delta vs. off | tok/s cost |
|---|---|---|---|
| Admission only (`moe_cache_atlas_warm`, original) | **4/4** | **+2.75pp** | -0.3% |
| Eviction-weighting only (the "corrected" redesign) | 3/4 | +0.63pp | -0.15% |
| Both combined | 1/4 | **-0.95pp (net negative)** | -0.84% |

Two real reversals in sequence, both reported honestly rather than kept as
the more optimistic earlier read: (1) the theoretically-cleaner eviction-
weighting redesign measurably underperformed the "redundant" admission
design it was meant to replace - right after a topic switch, most newly-
relevant experts genuinely aren't resident yet, so "protect what's already
there" has nothing to work with until something else brings them in first;
(2) combining both (on the theory that they're complementary, not
competing) measured *worse* than either alone, not better - best guess is
real interference, not addition: the eviction-weighting factor makes
topically-aligned candidates resist eviction everywhere, including when
`moe_cache_atlas_warm`'s own admission calls need to evict something to
make room for a *different* atlas-suggested candidate, so admission ends up
fighting itself.

**Final state**: reverted to admission-only (`moe_cache_atlas_warm`,
unchanged from the original design) - the clear, empirically-best performer
across all three variants tested. `moe_cache_atlas_align_weight` (the
eviction-weighting factor) was removed from the codebase entirely, not left
disabled - it measured worse standalone and worse combined, with no
condition under which it won. Still gated behind
`GGML_CUDA_MOE_CACHE_ATLAS_WARM`, off by default; the topic-switch result
(+2.75pp hit rate, 4/4 rounds, small tok/s cost) is real but has not yet
been re-validated with more rounds the way the steady-state tests were -
worth doing before considering a default flip.

## Step 6 — measured: is the co-activation signal actually predictive?

Built opt-in scoring (`GGML_CUDA_MOE_CACHE_MEASURE_PRED=1`, commit
`95456179c`) rather than building a policy on an unmeasured signal - this
session had already produced three mechanisms that were built, A/B'd, and
found flat or reverted.

**Result (Ornith-1.5-35B, 22000 scored predictions, 9475 tracked edges):**

| predictor | depth-1 precision | cost per layer |
|---|---|---|
| router-lookahead (existing) | **59.3%** | real `n_embd x n_expert` matmul |
| cross-layer co-activation | **33.8%** | hash lookup |

Stable as the table grew (33.8 / 33.7 / 33.0 / 33.8 over 16k-22k samples),
so converged, not undertrained. Worse than router-lookahead at depth 1, and
worse even than its depth-3 precision (41.1%).

Structural reason: the router predicts from the **actual hidden state of
the current token**; the co-activation table is a static frequency prior
with no token-specific information, and routing is data-dependent. A prior
cannot compete with a computation that sees the input.

**Consequence: step 7b (co-activation as a cheaper/deeper prefetch
predictor) is DROPPED, not built.** Measuring cost ~15 lines; building it
would have cost far more and ended flat.

**Step 7a - MEASURED, and it is strong** (commit `ea822e946`). Different
question from step 6: "given one expert in a routing decision, is its most
frequent partner also selected in that SAME decision?"

  within-layer partner precision@1:  **67.6%** (54080/80000 scored decisions)
  chance baseline:                     2.8%  (from real n_ids / n_expert)
  -> **24x over chance**, stable and rising slightly as the table matures
  (cross-layer prediction sat at 37.2% in the same run, for contrast)

This differs from step 6 in kind, not degree. Step 6 was *prediction*,
competing against a router that sees the token's actual hidden state and
therefore wins. 7a measures co-occurrence *structure*, which the router's
per-token computation does not help the cache exploit at all - the cache
still admits **one expert at a time**, so when A is admitted and B co-fires
67.6% of the time, that is a miss taken structurally, not for want of a
prediction.

**This is the first result in Track 1 that justifies building on it.**
Next: group-aware admission - admit/evict correlated experts as a unit
rather than individually - then A/B it like everything else. Not built yet.
Worth noting the honest risk before building: a 67.6% partner hit rate does
not automatically mean group admission wins, because admitting B early also
costs a slot that something else could have used. The A/B decides it, not
the precision number.

## Track 1.6 — Discovered topics from co-activation — ATTEMPTED, inconclusive (2026-08-25)

Built (`scripts/moe-discovered-atlas.py`): factor the co-activation graph, use
its components as axes, no category list at all. Global graph over all
(layer, expert) nodes rather than one embedding per layer — per-layer
factorings each come out in their own arbitrary rotation, and `req_dir`
averages positions across layers, so it needs one shared space. Randomised
subspace iteration over sparse matvecs, so 8,661 nodes never need a dense
factorisation. Output matches the existing atlas schema, so it is a drop-in.

Built on co-activation, NOT `req_dir`, which is the whole point — the
circularity argument below still holds. Verified in code that substitution
does not break it: `co_activation` is keyed on `ids[a]`/`ids[b]`, the
router's *requested* experts, so a substituted pick never feeds back.

**Held-out result** (graph from the first half of each trace, scored on the
second): the discovered axes carry essentially no topic information.

| space | Fisher | topic-acc (chance 25%) |
|---|---|---|
| human 9 categories (probe-measured) | 0.4136 | **71.6%** |
| discovered (raw) | 0.0005 | 25.5% |
| discovered (layer-centred) | 0.0030 | 31.0% |

**Diagnosis — the axes encode LAYER, not topic.** Fisher ratio for layer
identity is **2.5420** against 0.0005 for topic. A co-activation graph is a
set of per-layer cliques (experts only co-fire within a layer), so that is by
far its dominant structure and spectral factoring spends every component on
it. Not a cross-layer sampling artifact either: weighting all-pairs
cross-layer edges up to 64x moved layer-Fisher only 2.475 → 2.279, with topic
accuracy pinned at chance throughout.

**Why this is NOT yet a verdict.** Two objections, both fair:

1. *The data was far too thin.* 28,800 decisions over 4 topics gives a
   co-activation graph at 22.9% fill with **median edge weight 2, and 46% of
   pairs seen exactly once**. Half the graph is noise. Projection: median
   weight 20 needs ~10x the data (~1,800 tokens/topic), median 50 needs ~25x.
2. *One-shot spectral is not the mechanism described.* The design intent was
   experts that co-fire *moving nearer* and the map *settling over a long
   run* — incremental attraction dynamics, accumulated and persisted like
   `session.history`. Eigendecomposition solves a fixed objective on a fixed
   snapshot in one step; it has no notion of settling and no memory between
   runs. It is a different algorithm, not a fast approximation of that one.

What does survive, because it is about graph topology rather than counts: the
layer blocks are intrinsic and any method here must handle them explicitly.
The folds, if they exist, live *within* each layer with cross-layer alignment
stitching corresponding regions — not as one undifferentiated cloud, which is
what the global embedding assumed.

### Label-free rebuild on 31.6x the data — the discovered atlas still loses

Rebuilt on 909,480 pooled decisions from 12 topics (2,000 tokens each), with
the topic labels discarded entirely, and scored by held-out link prediction
(`scripts/moe-link-prediction.py`): given two experts in a layer, will they
fire in the SAME routing decision later? Positives are held-out pairs that
were **never seen in training** — predicting a pair already observed is
memorisation, not prediction. Negatives are sampled within the same layer, so
layer structure confers no advantage either way.

The data objection is now answered: median edge weight 4 (was 2), mean 16.0,
seen-once down to 21.8% (was 46%).

| method | AUC |
|---|---|
| popularity control (`log f_a + log f_b`) | 0.5437 |
| **probe atlas (9 categories)** | **0.7116** |
| discovered co-activation (16 dims) | 0.5706 |
| discovered co-activation (32 dims) | 0.5715 |
| discovered co-activation (64 dims) | 0.5761 |

The discovered embedding **does** beat the popularity control, so it has
learned something real rather than reflecting expert frequency — but it
saturates around 0.576 and under-parameterisation is not the explanation.

The uncomfortable part, stated plainly: **the probe atlas predicts co-firing
better than an embedding built from co-firing itself**, and it does so having
never seen this traffic. Topic affinity appears to be a better organising
principle for "which experts work together" than the co-firing graph's own
spectral structure. That is an argument FOR keeping the 9 categories, arrived
at by a test designed to be fair to the alternative.

**Still genuinely open**: the algorithm. Everything tested so far is one-shot
spectral factorisation of a snapshot. The design intent — co-firing experts
*moving nearer*, the map *settling over a long run*, persisted across restarts
like `session.history` — is incremental attraction dynamics and has not been
built. Spectral factoring is not a fast approximation of it. That remains the
untested hypothesis, and it is the one worth building next.

**Next**: rebuild on ~34x the data (12 topics x 2,000 tokens), pooled and
UNLABELED, and score by held-out link prediction
(`scripts/moe-link-prediction.py`) rather than by agreement with the 9
categories — scoring "did it recover our labels" is rigged toward the probes.
The popularity control in that script is the one that matters: frequently-used
experts co-fire often merely because they are frequent, so any embedding that
cannot beat popularity has not earned its complexity.

## Track 1.6 — original note (circularity argument, still stands)

The atlas's 9 categories are human guesses hardcoded in `k_probes[]`, and a
*dynamic* atlas built on them has a trap: live traffic carries no category
label, so the only available labeller is `req_dir` — itself inferred from
the static atlas. Updating the atlas from that is self-training, and the
known failure mode is confirmation drift (a slightly wrong estimate
reinforces itself).

Co-activation (steps 4a/4b, already collected) sidesteps both problems at
once. "These two experts fired in the same routing decision" is a direct
observation, not an inferred label, so there is no circularity — and
embedding/clustering that graph yields dimensions that are *discovered*
rather than assigned, including structure no human would have thought to
name. One mechanism answers both "can the atlas be dynamic" and "must it
use preset topics", with no damping leash required.

Sketch: build an expert-by-expert co-activation matrix, factor or cluster it
(spectral / NMF / graph embedding), use the resulting components as axes.
The `cats{}` vector now emitted by `llama-expert-atlas` makes the two
directly comparable — discovered axes can be scored against the labelled
ones to see how much of the human category set the model actually
reproduces on its own. Not started; needs its own design pass.

If instead staying with labelled probes, the cheap intermediate is simply
more categories: `cats{}` is schema-stable under a changing category set,
and the sphere anchors are computed for any N, so growing `k_probes[]` and
regenerating needs no code change on either side. Untested — no measurement
has been taken of whether more categories actually improves anything.

## Track 1.5 — CPU-GPU split gap (from FreeToken, never carried over)

Real gap identified while re-testing step 3, not hypothetical: the atlas
warming ranking pass (`moe_cache_atlas_warm`'s cosine-similarity scan over a
tensor's whole atlas-covered candidate set) runs **inline, synchronously, on
the critical GPU-dispatch path** (`moe_cache_plan`, which must complete
before the GPU op for that layer can be issued) - not offloaded to the
existing background fill-worker thread the way the actual expert-weight
*copies* already are. FreeToken's reference architecture keeps this kind of
CPU-side prediction/ranking work off the hot dispatch path, running on spare
CPU cycles while the GPU is busy with the previous layer's compute, instead
of blocking the thread that has to issue the next GPU op. We didn't carry
that split over when building Track 1 - everything CPU-side (the ranking
loop, not just the fill worker's actual copies) has been synchronous so far.

**Hypothesis, not yet measured**: moving the ranking computation itself onto
the background worker (leaving only "read the precomputed result and admit a
slot" on the critical path, the same shape moe_cache_plan already uses for
real demand fills queued to the worker) could reduce the per-call latency
atlas warming currently adds every 64 plan() calls, independent of whether
the warming decision itself is good. Not started - needs its own scoped
design, likely reusing the existing `device.queue`/worker-thread plumbing
rather than inventing new infrastructure.

**BUILT (2026-08-24), NOT YET MEASURED.** The split now exists, off by
default behind `GGML_CUDA_MOE_CACHE_ATLAS_WARM_ASYNC=1` (and only reachable
at all with `GGML_CUDA_MOE_CACHE_ATLAS_WARM=1`, which is itself off by
default). It reuses the existing fill-worker thread and session lock; no new
thread, no new synchronization primitive.

What changed:

- `moe_cache_atlas_warm` was split into its two real halves.
  `moe_cache_atlas_rank` is the expensive one - a cosine scan over the
  tensor's whole atlas-covered candidate set, a `sqrt` and a few flops each,
  up to `n_expert` of them - and is now *pure*: it touches no cache state and
  takes no lock. `moe_cache_atlas_admit` is the cheap one - a few hash
  lookups, at most one eviction, a queue push - and is the only half that
  needs `session.mu`.
- Async path: `moe_cache_plan` now does a bounds check, a hash lookup and a
  `push_back` (`moe_cache_atlas_warm_enqueue`), and nothing else. The fill
  worker picks the request up in its existing wait loop, **releases the
  session lock for the ranking scan**, retakes it, re-validates the pool, and
  admits. Ranking therefore overlaps whatever the GPU is doing, which is the
  FreeToken split this port never carried over.
- The warm queue is deliberately **one deep**. A warming hint describes where
  the request is heading *now*; if one is still pending, a second is already
  stale by the time it would run, and queueing it would only make the worker
  do the same scan twice. Dropping is correct, not a shortcut.
- `atlas_by_tensor` rows are now `shared_ptr<const ...>` and `set_atlas`
  publishes a replacement row instead of mutating in place. Without that, a
  re-registration could free the vector the worker is mid-scan on. The
  ranking pass also captures `req_dir_x/y` **by value** at enqueue time, so
  it ranks against where the request was heading when the hint was raised,
  not wherever the centroid drifted to by the time the worker got there.
- The sync path (default) still runs both halves inline under the lock, so
  the old behavior is what ships and the two are directly A/B-able.

**One real behavior change, not hidden:** the old scan skipped
already-resident experts inline, which needs `pool.map` and therefore the
lock. The lock-free pass cannot, so it keeps a deeper list
(`MOE_CACHE_ATLAS_RANK_K=4`) and the admission half - which holds the lock
and already re-checked residency anyway - walks down it admitting the same
top-2 non-resident candidates in the same score order
(`MOE_CACHE_ATLAS_ADMIT_K=2`). The two differ only when 3+ of the top 4 are
already resident. Small, but it means the +2.75pp topic-switch number above
was measured with the *old* scan and would need re-validating before any
default flip.

**Measurement hook, not yet run**: `GGML_CUDA_MOE_CACHE_ATLAS_WARM_TIMING=1`
accumulates the time the warming block costs *on the dispatch path
specifically* and prints a running average every 256 warm calls. That is the
quantity this track is about, and tok/s is the wrong instrument for it - the
block fires once every 64 `plan()` calls, so even a large per-call saving is
a rounding error at the throughput level. Measuring the thing directly, for
the same reason steps 6 and 7a were measured before anything was built on
them.

### MEASURED (2026-08-24) - the hypothesis holds, and the caveat matters more

Ornith-1.5-35B Q4_K_M, `--moe-cache auto -c 65536 -ngl 99`, its own matched
atlas, `GGML_CUDA_MOE_CACHE_ATLAS_WARM=1`, scratch server on 8098 (the live
8099 deployment was stopped for the test and restored to its exact
last-known-good command immediately afterward). Two interleaved rounds per
arm, four 700-token generations each (alternating a code-heavy and a
medical prompt), timing read from
`GGML_CUDA_MOE_CACHE_ATLAS_WARM_TIMING=1`.

| | dispatch-path cost | hit rate | prefetches | evictions |
|---|---|---|---|---|
| sync (old, default) | **4.9 us** / 4.9 us | 77.60% / 77.53% | 3591 / 3584 | 53015 / 53268 |
| async (Track 1.5) | **0.6 us** / 0.5 us | 77.67% / 77.71% | 3592 / 3623 | 52808 / 52535 |
| | **-88%** | +0.13pp | +0.6% | -0.9% |

Both arms coherent, zero `CUDA error` / `ggml_abort` / copy-failure-disable
markers across all four servers. The per-arm averages were flat from the
first 2048-call print to the last (4.8-4.9 vs 0.5-0.6), so this is converged,
not a warmup artifact - the cleanest signal since the D2D measurement, and in
the same style: near-zero within-arm variance, no overlap between arms.

**What it means, stated narrowly.** The ranking scan really was ~4.3 us of
synchronous CPU work sitting on the GPU-dispatch path, and moving it to the
worker removes essentially all of it - what remains (0.5-0.6 us) is the hash
lookup and `push_back` the enqueue still has to do. The cache does the *same
work*: prefetch counts land within 1% and hit rate within 0.2pp, i.e.
deferring the ranking by one worker hop did not make it rank against a
meaningfully staler direction, which was the main correctness worry.

**What it does NOT mean.** This is not a throughput win and should not be
reported as one. tok/s was 58.6-62.0 in both arms with full overlap
(59.72/61.93/58.61/60.97 sync vs 59.11/62.00/59.25/61.32 async) - exactly as
expected, because 4.3 us saved once every 64 `plan()` calls is ~0.07 us per
call against a ~17000 us token, i.e. four orders of magnitude below anything
tok/s can resolve. The honest summary is: **the latency this track set out to
remove is real and is now gone, and it was never large enough to matter for
throughput at the current 1-in-64 rate limit.** That is a completed
architectural fix, not a performance result.

Where it *would* start to matter: any future change that raises the warming
rate (a lower rate limit, per-layer warming, a larger candidate set, or a
richer atlas with more categories per expert - all of which scale the scan,
none of which now scale the dispatch path). The split is what makes those
affordable to try; it is not itself a win to bank.

Still off by default. Nothing here re-validates the +2.75pp topic-switch
result, which was measured with the old residency-filtering scan (see the
behavior-change note above) - that remains the open question before any
default flip, and this measurement deliberately did not touch it.

## Track 2 — Unified tiered placement — RESCOPED (mostly already exists)

Original framing was "remove the hard `-ncmoe` boundary, give every MoE layer
one global LFRU budget" - loader change, budget-math change, cache-sizing
change. Investigating the code before building showed most of that is
already true:

1. **Pools are keyed by `(expert_size, wtype)`, not by layer**
   (`moe_cache_find_pool`). A single global budget spanning all layers
   already exists - every layer sharing a tensor shape shares a pool.
   Nothing to build.
2. **Offloading every MoE layer is already expressible** as
   `-ncmoe <n_layer>`; `common_moe_build_cpu_overrides(n)` simply emits an
   override per layer below `n`. No loader change needed.
3. What genuinely remains is a **default-policy question**, not an
   architecture one: `common_maybe_raise_moe_for_ctx` picks the *minimum*
   offload that makes the requested context fit. Its own comment already
   flags this as "a sufficiency question, not a throughput one", with a
   measured counter-example (Nemotron 3.5: fit search stopped at 46 layers
   -> 1239 slots -> 57.1% hit rate; offloading all 53 -> 2307 slots ->
   68.3%). `--moe-calibrate` already finds the throughput optimum
   empirically, but is opt-in.

**So the remaining work is an experiment, not a rewrite**: compare
(a) auto placement (minimum that fits) against (b) `-ncmoe <n_layer>`
(maximum offload, larger cache budget) on tok/s and hit rate, on this
hardware and model. If (b) wins consistently, the change is to the default
placement policy - a small, well-contained edit - not to the loader or the
cache. Testable with zero new code.

Caveat worth stating before running it: more offload means more experts
served from the cache, which is only a win while the cache can hold enough
of the working set. Past some point it trades resident weights for cache
capacity that cannot cover the extra demand, and the constrained-regime
result (33% hit rate at a 512 MiB budget) shows what the far end of that
looks like. The optimum is empirical, which is exactly why
`--moe-calibrate` exists.

**Status (2026-08-24): NOT RUN.** The rescope (`2b721d844`) is the last thing
that happened here - it established that items 1 and 2 need no code, which is
what makes this track look further along than it is. The one thing actually
left, the (a)-vs-(b) placement comparison, has not been measured: no commit,
no numbers, nothing in `docs/index.html`. It needs no code at all, only the
GPU - and the same 8099 occupancy blocking the Track 1.5 A/B blocks it too.
Track 2 is therefore *descoped*, not *done*.

**MEASURED, 2026-08-29 - Track 2 is now actually done.** Ran the (a)-vs-(b)
comparison: auto placement (default `-fit`, no `-ncmoe`) vs forced full
offload (`-ncmoe 41`, every MoE layer), Ornith-1.5-35B, same topic-switch
6-prompt sequence and 4-round methodology as every other benchmark this
session, `-c 4096` both arms, only the placement flag differs.

| | auto (default) | full (`-ncmoe 41`) |
|---|---|---|
| avg tok/s (n=24) | **53.96** | 32.82 |
| avg hit rate (n=24) | 70.15% | **86.76%** |

Full offload gets a real, consistent +16.6pp hit rate - and is 39% *slower*
in actual throughput, every round, no exceptions. This inverts what the
Nemotron 3.5 counter-example above seemed to suggest, because that number
was hit-rate-only and never measured tok/s: forcing every layer through
moe-cache's dispatch path costs more in aggregate overhead than the better
hit rate saves. Visible directly in the per-request log: `full`'s tok/s
degrades over the course of each round (42->50->36->32->22->19, round 1)
as the larger pool keeps churning through admissions; `auto`'s stays
stable-to-improving (44->50->56->60->59->61) - the smaller pool converges
fast and stays converged.

**Verdict: the existing default (auto/minimum-fit placement) is correct
and should stay the default.** No code change needed - this closes Track 2
with a confirmed "no change required," not a regression to fix. Raw
results: `track2_bench_results.json` in this session's scratchpad.

## Track 3 — "True MoE" emulation

Newest addition, **not yet scoped**. Needs a clarifying conversation before
a real plan exists, because two very different projects both fit the name:

- MoE-izing the *dense* attention layers themselves (a genuine "mixture of
  attention experts" architecture change) — this is the only way the earlier
  "attention weights, only for the agent using them" table proposal becomes
  literally buildable, since standard dense attention has no per-token
  selection mechanism to hang caching off of.
- Using the *existing* MoE routing plus Atlas/co-activation data (Track 1) to
  treat clusters of frequently-co-activated experts as emergent "sub-models"
  within the current architecture — no architecture change, a different lens
  on data we're already collecting.

## Degenerate output under VRAM exhaustion — ROOT-CAUSED AND FIXED (2026-08-24)

**Root cause found and fixed** (`b321b6259`).
`moe_cache_prefill_advance()` published a prefill slot
(`inflight_host[slot] = next_host_base`) **unconditionally**, even when the
copies into that slot had failed. `prefill_wait()` uses `inflight_host` to
decide the buffer holds live data, so it then handed
`ggml_cuda_mul_mat_id` a buffer still containing the previous tenant's
bytes - read as expert weights. Every copy *was* error-checked; the return
values were discarded. Under VRAM pressure that is exactly how a cache
turns into a source of fluent-looking nonsense with a clean log.

Fixed by making `copy_split` report whether every copy landed, refusing to
publish a slot whose copy failed (so `wait()` returns NULL and the caller
falls back to its normal H2D path), and disabling the cache entirely after
N consecutive copy failures with a loud one-time log. Also removed the
`cudaErrorMemoryAllocation` exemption from `moe_cache_cuda_ok`'s fatal
branch - it was meant for retryable allocation sites but also covered OOM
on copy paths, where nothing retries and the buffer keeps stale bytes.

Historical detail retained below, including two wrong turns.



A live server returned **pure `////////` for every token**. Root cause is
almost certainly **VRAM exhaustion**, not any of this session's changes -
the user reports seeing it before the atlas work existed at all.

**Trigger correlates with VRAM pressure from a co-resident GPU process**,
in every observed case:
- first sighting: server started immediately after `llama-expert-atlas`
  had been holding the GPU
- second sighting: a 64k server running while diagnostic servers were
  launched alongside it
- a deliberate contention test: free VRAM already at 152 MiB, dropping to
  43 MiB once a second instance loaded - the request then hung outright
- every **single clean instance** test came back coherent, including
  `--moe-cache auto` *with* the atlas file (3/3 runs)

**Only reproducible through the chat-templated prompt**, not a raw one:
`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`.
A plain `/completion` prompt stays coherent even in a failing instance,
which is why an early pass at this wrongly concluded "not reproducible" -
it was testing the wrong endpoint. A later pass wrongly blamed the atlas
registration; a clean single instance with the atlas is fine.

Ruled out: Ornith itself, the chat template, the atlas file (v1 and v2),
context size alone, and moe-cache alone. Each is fine in isolation.

**The real defect is the failure mode, not the OOM**: moe-cache sizes its
slab from free VRAM at init, and when a co-resident process takes VRAM
afterward, it degrades into silently producing wrong weights - garbage
tokens - instead of raising a clean allocation error. The corrupted
instance logged no CUDA error, no fill failure, no dispatch failure, and
also hung on shutdown (ignored SIGTERM ~12 min, needed SIGKILL).

**Not fixed.** The worthwhile fix is defensive: detect allocation
failure/VRAM pressure after init and fail loudly (or fall back to
`--moe-cache off`) rather than emitting garbage. Practical workaround
today: run one model instance per GPU and check `nvidia-smi` for
competing processes before blaming the model.

## MTP speculative decoding: works, but NOT on Ornith (2026-08-24)

Prompted by the Codacus video's claim that the expert cache is worth ~5%
alone and that its real value is the interaction with speculative decoding.
Tested that premise on our stack; it does not transfer, and turned up a
separate real finding.

**Ornith-1.5-35B (`qwen35moe`, 41 blocks, 256 experts,
`nextn_predict_layers=1`) with its built-in MTP head is a large
regression**, reproduced across two rounds:

| config | tok/s | draft acceptance |
|---|---|---|
| cache off, MTP off | 47.28 / 46.91 | - |
| cache auto, MTP off | 65.33 / 65.87 | - |
| cache off, MTP on | 24.54 | 0.203 |
| cache auto, MTP on | 22.21 | 0.187 |

Control on the same build, Gemma-4-26B with its SEPARATE MTP draft model
(`mtp-gemma-4-26B-A4B-it-Q8_0.gguf`): acceptance **0.718 / 0.738**. So the
`--spec-type draft-mtp` path is fine - MTP wires up correctly (zero
"unused tensor blk.*nextn" lines once the flag is passed, versus 4
without). **Ornith's built-in head simply produces bad drafts**: ~0.19
acceptance at mean draft length 1.6 means paying a draft pass plus a
verify pass and discarding 4 of every 5 tokens, which is exactly the ~2x
slowdown observed. Most plausible cause: Ornith-1.5 is a finetune whose
`nextn` head was not retrained alongside the modified backbone, so it
predicts for a model that no longer exists. Not fixable in llama.cpp - a
property of the checkpoint.

**Actionable**: do not use `--spec-type draft-mtp` with Ornith. Nothing in
the output warns you; the failure is silent and expensive. A diagnostic
that fires when acceptance stays very low would be worth adding.

**The Codacus premise does not transfer.** They report cache alone at
+5% (42 -> 44 tok/s). On our stack the cache alone is worth **+38%**
(47.28 -> 65.33 on Ornith), so we are not leaving the cache x spec
interaction unexploited - we already get most of that value from the
cache by itself. That also explains why every cache-POLICY tweak measured
flat today: the cache is already doing its job here, and the remaining
headroom inside it is small.

**Not established**: whether Gemma+MTP is a throughput win on this rig.
Round 1 showed +43%, round 2 showed +1.5%, at essentially identical
acceptance - so the acceptance gap is solid but the throughput claim needs
more rounds before it means anything.

## Silent corruption fix — VALIDATED (2026-08-24, overnight)

The fix in `b321b6259` is now proven against a deterministic reproduction,
not just assumed. Getting there corrected two of my own wrong conclusions.

**Wrong conclusion 1 (mine): "the prefill double-buffer is inert in server
mode."** It is not. There are TWO separate prefetch mechanisms and I was
enabling the wrong one for hours:
- `GGML_SCHED_PREFETCH_EXPERTS` - the ggml-backend scheduler's prefetch
  slots (the +12.7% pp2048 measurement)
- `GGML_CUDA_MOE_PREFILL_BUFFER` - moe-cache's own prefill double-buffer
  (`register_successor` / `advance` / `wait` / `copy_split`), which is
  where the corruption fix lives

With the correct flag the path is healthy: **122 HIT / 40 MISS**.

**Wrong conclusion 2 (mine): "fault injection proves the fix is
incomplete."** That run showed `////////`, but it was never exercising the
injected path (wrong flag), so it was the original intermittent bug
appearing spontaneously - not evidence against the fix.

**Validation, Gemma-4, `GGML_CUDA_MOE_PREFILL_BUFFER=1`, fault injection
every 3rd copy:**

| | no injection | injection |
|---|---|---|
| prefill HIT | 122 | 19 |
| prefill MISS | 40 | 143 |
| output | OK | **OK 3/3** |
| loud disable | - | **fired** |

Hits collapse (failed copies are no longer published), callers fall back,
output stays correct where this path previously produced `////`, and the
safety valve logs once: `CUDA0 DISABLED after 8 copy failures (likely VRAM
exhaustion) - falling back to uncached offload`.

**Still true and still worth fixing separately**: only `gemma4.cpp` calls
`build_moe_prefill_prefetch`, so the moe-cache prefill double-buffer (and
its LFRU **D2D** path) is unavailable to every other architecture,
including Ornith (`qwen35moe`). That is a real coverage gap, not a bug in
the mechanism.

## Prefill double-buffer coverage gap — attempted, REVERTED (2026-08-24)

Only `gemma4.cpp` calls `build_moe_prefill_prefetch`, so the moe-cache
prefill double-buffer and the LFRU **device-to-device** path riding on it
(a cache-resident expert copied D2D instead of over PCIe) are unavailable
to every other architecture, including Ornith (`qwen35moe`). Measured
before any change: **0 prefill hits on qwen35moe** (162 consulted, 162
miss) vs **122 hits / 40 misses on gemma4**.

Tried closing it by adding the same registration to `qwen35moe.cpp`.
**Registration works** - hits went 0 -> 14 (of 48 consulted) - **but the
model then crashes**:

```
prefill wait-for-consumed failed: operation not permitted on an event
last recorded in a capturing stream
CUDA error: operation failed due to a previous error during capture
-> ggml_abort
```

`moe_cache_prefill_advance` does a host-side `cudaEventSynchronize` on the
slot's `consumed` event before overwriting the buffer. That call is illegal
while a CUDA graph capture is active on the stream that recorded the event.
Gemma's capture boundaries evidently avoid this; qwen35moe's do not.

**Reverted** - a crashing change is worse than a missing optimization, and
this is not something to leave in the tree unattended. Verified clean after
revert (correct output, zero crash markers).

**So the coverage gap is not an oversight - it is gated on a real
capture-safety problem.** Closing it properly means making the
wait-for-consumed step capture-safe, e.g. skipping the host sync when a
capture is active and relying on stream ordering instead, or restructuring
so the overwrite cannot race the consumer. That is a careful, standalone
piece of work with a genuine correctness hazard (the wait exists to stop a
prefetch two calls later from overwriting a buffer still being read), and
should not be attempted casually. This is the natural next task for anyone
picking Track 1 back up, and it is the prerequisite for the D2D-instead-of-
PCIe idea on any non-Gemma model.

## moe-cache prefill double-buffer (the D2D path) — MEASURED, costs 5%

Never benchmarked before tonight: the documented +12.7% belongs to the
SEPARATE `GGML_SCHED_PREFETCH_EXPERTS` mechanism, not to this one. Gemma-4
is the only architecture that registers prefill successors, so it is the
only place this could be measured at all.

`llama-bench -p 2048 -ncmoe 15 --moe-cache auto`, 3 rounds:

| `GGML_CUDA_MOE_PREFILL_BUFFER` | pp2048 | values |
|---|---|---|
| off | **953.00** | 954.8, 951.8, 952.4 |
| on (D2D active) | 905.35 | 906.3, 905.0, 904.8 |
| | **-5.00%** | 3/3 rounds |

Near-zero variance within each arm - the cleanest signal measured in this
whole effort, and unambiguous.

**So the D2D path does not pay, even where it works.** D2D genuinely avoids
PCIe for cache-resident experts, but the double-buffer machinery wrapping
it - per-layer slab bookkeeping, host-side event syncs, `advance`/`wait`
lock traffic on every dispatch - costs more than the PCIe traffic it saves
at this scale.

**Direct consequence: do NOT do the CUDA-graph capture-safety work** needed
to extend this to `qwen35moe`. That effort would only make a measured
regression available to more architectures. The coverage gap documented
above is therefore not worth closing as things stand; it would need the
per-dispatch overhead reduced first, and only then re-measured.

## Resident-constrained routing ("use what is in VRAM") — 2026-08-25

Design premise: stop racing to fill VRAM in time, and instead run whichever
experts are already resident. Measured end to end this session.

### The premise is correct, and here is the measurement that proves it

Removing the admission gate entirely (`ADMIT_AFTER=1` + `THROTTLE=1`) buys
**+10.79pp hit rate and costs -7.6% throughput** (52.43 vs 56.75 tok/s,
evictions 110,169 vs 23,714). The racing *succeeds* at filling VRAM and loses
anyway, because fill traffic is the binding constraint. `ADMIT_AFTER=1` alone
is ~neutral (+0.6 tok/s), because `readmit_after` defaults to **8** and
`max(admit_after, readmit_after)` is the gate that actually binds for anything
previously evicted.

Corollary: **hit rate is purchasable and therefore not a valid objective on
its own.** 79.25% is available for one env var and a slower server.

### Prefetching is capped, not merely weak

Atlas warming gains **+0.31pp** hit rate in simulation. A *topic oracle* with
perfect foreknowledge of the segment's experts gains **+0.30pp**. The atlas is
already at the ceiling; the ceiling is the problem. Cause: every topic touches
~6,000 of 7,680 experts, with top-1000 covering only 52-61% of selections —
there is no compact topic hot set to preload. This single result explains
every refuted Track 1 A/B at once.

Atlas warming on/off on the live server, 2 reps each: **+0.27 tok/s, +0.71pp
hit, +509 fills out of ~23,000** — within run-to-run spread, i.e. neutral to
marginally positive. It is rate-limited to once per 64 `plan()` calls and
admission-gated, so it never generates enough fill traffic to compete with
demand. It is not free-loading.

### Where the headroom actually is: eviction

Belady's optimal beats LRU by **+16.86pp** at the real 18.5% cache ratio
(81.85% vs 64.98%). Cross-check: the live server measures 68.7% hit on
topic-diverse traffic against the LRU simulation's 64.98%, so the production
policy is behaving near-LRU here and that headroom is real.

Five signals tested as eviction policies (`scripts/moe-eviction-sim.py`), all
scored on retained gate mass, sampled eviction K=32:

| policy | retained | hit |
|---|---|---|
| **lru** | **75.72%** | 58.63% |
| random | 72.51% | 53.31% |
| coact | 72.74% | 58.12% |
| atlas | 71.06% | 51.59% |
| router score | 68.90% | 46.27% |

**LRU wins on retained mass at every fill rate**, and atlas and router score
lose to *random*. Both rank experts context-free, with no knowledge of what
was just used, so evicting by them tears up the working set recency is
holding. Belady is recency with perfect foresight — so the 17pp lives in
predicting **reuse distance**, and none of the available signals approximates
it. Co-activation is the sole exception: it beats LRU on **hit rate** at fill
rates above 1 (64.88% vs 63.37% at fill 4), reaching a higher hit rate with
*fewer* fills.

### Fill cost — and a retraction

Both simulators originally scored hits and gate mass with **no notion that
admitting an expert costs anything**, and consequently recommended "admit
everything" — the worst arm on real hardware. `scripts/moe_cost_model.py`
fits tok/s to three measured arms rather than inventing constants:

```
tok/s = -42.290 + 153.014*hit - 140.862*fills_per_pick
  1pp of hit rate  = +1.5301 tok/s
  1 fill/1000 picks = -0.1409 tok/s
  break-even: 1pp of hit must not cost more than 10.9 extra fills/1000 picks
```

Distinct from the cache's existing `MOE_CACHE_COST_TIER_NVME/RAM` weight (5.8
vs 1.0), which ranks *which* expert is dearer to refetch, not the cost of
filling at all. Applying it inverts the earlier conclusions: fill_rate 1 beats
fill_rate 8, and the corrected model reproduces the ordering it previously got
backwards ("admit everything" now ranks last, 10.34 vs 30.17). Absolute
predictions are NOT trustworthy — the sims run at 52-450 fills/1000 picks
against a calibration range of 40-188. Ranking only.

### Substitution — BUILT, measured, off by default

`GGML_CUDA_MOE_CACHE_SUBSTITUTE=1`. On a miss, run the best resident expert of
the same tensor instead of the CPU fallback. Mechanically only needs
`slot_indices[]` to point at a different slot; the gate weight applied
afterwards is still the one the router computed for the expert it asked for.

Stand-in chosen by **co-activation** — how often the candidate fired in the
same routing decision as the expert it replaces. Candidates enumerated from
the resident bitmask (one hash for the tensor, then bit tests over ~4 words),
which is why that structure exists: 64 hash lookups per decision on the
dispatch path is what Track 1.5 exists to avoid.

Admission bookkeeping deliberately still follows what the *router wanted*, not
what was executed, or resident experts become the only ones ever used and
therefore the only ones ever kept. A stand-in earns heat but never promotion
to protected.

| | tok/s | hit | substitutions |
|---|---|---|---|
| OFF | 56.82 | 67.62% | 0 |
| ON | **59.88** | 65.51% | 200,637 (99.5% of misses) |

**+5.4% throughput.** Quality, paired per-chunk over 120 chunks (the unpaired
±1.5% bars cannot resolve this; pairing tightens it 11x):

```
mean   +0.181% perplexity
95% CI [-0.106%, +0.468%]     t = 1.23, NOT significant
chunks where ON was worse: 59/120 (49%)
```

So the cost is **bounded at +0.47% perplexity at 95% confidence** for +5.4%
throughput. Caveats: one corpus (this repo's docs), and perplexity is a weak
proxy — it will not catch degradation in long-form reasoning at a 99.5%
substitution rate. Keep off by default until run against a reasoning
benchmark.

Measuring this needed `-b 4 -ub 4`: co-activation is only recorded when
`n_ids <= 32`, so any larger batch leaves the substitution table empty and
both arms return byte-identical PPL. Cost two wasted runs to rediscover.
It also means **substitution is inherently decode-only**.

### Instrumentation added

- **Rank-resolved hit/miss** (`rank_hits`/`rank_misses` on `/experts`): misses
  concentrate in the router's low-confidence tail — rank 0 misses 20.0% of the
  time, rank 7 misses 44.0%, monotonic, with only **8.0% of all misses landing
  on the router's top pick**. That is what makes substitution cheap.
- **Signal 3** (`MOE_TRACE_SCORES=1` in `llama-moe-trace`): the router's score
  per pick, not just its ranking. Rank 0 carries 26.48% of gate mass, rank 7
  carries 6.82% (3.88x spread); ranks 4-7 together carry 32.17%.
- **Resident bitmask** (`GGML_CUDA_MOE_CACHE_MASK_AUDIT=1` proves it): one bit
  per expert per tensor, maintained only through
  `moe_cache_map_insert`/`erase`. Validated live with the pool saturated and
  13,639 evictions of churn: 8 audits, 0 disagreements.

### Hit rate is workload-dependent — quote it with its traffic

A single repeated prompt measures **100.00%** (0 misses in 601,440 lookups).
Four topic-diverse prompts measure **68.7%**. Same cache, same size. Topic
diversity produces misses, not cache pressure — so any hit-rate figure quoted
without its workload's topic spread is meaningless.

## Anti-co-firing and the incremental atlas — both tested, both fail (2026-08-25)

### Anti-co-firing exists, and is stronger than attraction

Measured over 70,800 frequent same-layer pairs (expected co-fires >= 5), on
909,480 pooled decisions:

```
NEVER co-fire despite that expectation : 6.68%
PMI mean -1.123   median -0.784
share negative (anti-associated)       : 71.5%
share strongly negative (PMI < -1)     : 45.0%
share strongly positive (PMI > +1)     :  8.2%
```

Strong repulsion (45%) is 5x more prevalent than strong attraction (8.2%).
Every mechanism built so far uses only the attraction half.

*Correction*: a first pass computed expected co-fires as `pa*pb*N*(K-1)`. The
extra `(K-1)` inflates the expectation 7x and biases every PMI down by
log(7)=1.95, which produced a spurious "98.6% anti-associated". Correct
expectation under independence is `N*pa*pb`.

### As an eviction signal it is the worst arm yet

1421 slots, window 64, all scores accumulated online, 151,580 decisions:

| fill | policy | retained | hit | fills/1k | pred tok/s |
|---|---|---|---|---|---|
| 1 | **lru** | **89.69%** | 77.45% | 92.4 | **63.20** |
| 1 | random | 86.37% | 71.09% | 106.9 | 51.43 |
| 1 | coact | 82.70% | 72.98% | 88.2 | 56.95 |
| 1 | anti | 73.76% | 59.11% | 108.2 | 32.91 |
| 1 | anti+lru | 74.71% | 60.26% | 106.9 | 34.85 |

Worse than random, and it generates MORE fills. The reason is structural and
general: anti-co-firing measures **simultaneity** ("not in the same decision"),
while eviction needs **temporal proximity** ("not needed soon"). An expert can
be strongly anti-associated with the current picks and fire on the very next
token.

**That is now six signals beaten by plain LRU** — atlas, router score,
co-activation, frequency, admission demand, anti-co-firing. Every
co-activation-derived signal describes within-decision relationships in one
polarity or the other; eviction is an across-time question. The single place
such a signal has ever won is SUBSTITUTION, a placement decision, where it
delivered +5.4%.

This supports separating the two roles explicitly: co-firing structure is a
hard rule governing PLACEMENT (who stands in for whom, who sits near whom),
and recency/heat is the soft rule governing RETENTION probability. Applying a
placement signal to a retention decision has now failed six times.

### Incremental atlas with AdaGrad, and the SILO WARNING (2026-08-25)

Per-node AdaGrad replaces uniform averaging: each node's step scales by its own
accumulated gradient history, so frequent experts damp down as their history
grows while rare ones keep moving — "keep the history of affinity rather than
overwrite it", applied per node. Base learning rate still constant.

Dynamics are now correct: 89.9 deg -> 7.7 deg over 14 epochs, still decreasing,
no oscillation, no collapse (pairwise cosine sd 0.177), and no edge-sticking
(mean radius 0.398 in the 2D projection, 5.1% in the outer 20%). Persistence
verified: resuming adds to the saved map (152.8M -> 178.3M steps) and settling
continues smoothly rather than restarting.

Held-out link prediction is still **AUC 0.4823 — chance** (0.4991 under uniform
averaging). AdaGrad fixed the dynamics, not the prediction.

**But the isolated test does not predict integrated behaviour.** Loaded onto
8099 against the probe atlas, with prompts absent from all 12 trace topics and
every earlier A/B, 2 reps each:

| atlas | tok/s | hit | fills |
|---|---|---|---|
| probe (9 categories) | 58.45 | 71.20% | 21,714 |
| incremental (AUC 0.48) | 58.43 | 71.28% | 20,875 |

**Identical.** A map scoring at chance performs exactly as well in the running
system as one scoring 0.7116. Cause: atlas warming is rate-limited to once per
64 `plan()` calls and admission-gated, contributing ~2% of fills, so the system
barely consumes the atlas — consistent with the warming on/off result
(+0.27 tok/s, within noise).

Consequence for how this is evaluated: **the probe atlas's link-prediction
advantage is real as a statement about the maps, but it never reaches the
system.** No atlas comparison run through 8099 can currently distinguish a good
map from a bad one, because the consuming mechanism has almost no leverage.
Any future claim that one atlas beats another must say which of the two it
means. The incremental map did produce 3.9% fewer fills, consistently across
both reps, which is the direction fill cost rewards.

Untested suspicion: training negatives are drawn uniformly from same-layer
experts while the graph is 22.9% dense, so ~a quarter of the "negatives" are
real co-firing pairs. That would poison SGNS and could explain chance-level AUC
despite clean convergence.

### Incremental atlas — first attempt, settles but learns nothing

`scripts/moe-incremental-atlas.py`: SGNS over co-firing pairs, constant
learning rate by design (annealing would manufacture the appearance of
settling), state persisted to .npz and resumed on the next run so it
accumulates like `session.history`.

Two failure modes found and fixed in order: (1) unnormalised vectors diverge
to 1e70 within 4 epochs; (2) with normalisation they OSCILLATE, flipping to
the antipode every pass (median move 2.0 on a unit sphere), because `np.add.at`
applies every update a node collected in a batch at once, so a frequently
selected expert takes a step proportional to its frequency. Averaging each
node's update by its occurrence count fixes both.

It then settles cleanly — 0.25 deg/epoch, stable from epoch 2, and does NOT
collapse (pairwise cosine mean -0.0013, sd 0.1767, range -0.69..+0.65, so
negative sampling supplies real repulsion). But held-out link prediction gives
**AUC 0.4991 — pure chance**. It settled into something with no predictive
content: the per-node averaging that cured the oscillation shrank each step
enough to underfit. That is an implementation failure, not a refutation of the
incremental idea, and it remains the one untested variant.

Reference points on the same test: probe atlas 0.7116, spectral discovered
0.5761, popularity control 0.5437.

*Note*: with 5x the data LRU's absolute retained mass rose from 75.72% to
89.69%. Rankings held throughout, but earlier absolute simulation figures are
pessimistic.

## THE SPEED AND THE CORRUPTION ARE THE SAME MECHANISM (2026-08-25)

Measured, 4 prompts per arm, conservative detector:

| config | tok/s | degenerate |
|---|---|---|
| `--moe-cache off` (safe) | 44.4 | 0/4 |
| cache 3072, `ADMIT_AFTER=32` | **64.6** | **3/4** |
| cache 3072, `ADMIT_AFTER=128` | 59.4 | 3/4 |
| cache 3072, `ADMIT_AFTER=255` | 43.0 | 0/4 |

**There is no middle ground.** Throttling admissions trades speed back linearly
until "no admissions", which performs the same as disabling the cache. The
~65 tok/s this deployment used to see was the CORRUPTING configuration - the
throughput and the fault come from the same mechanism, so no setting today
delivers 65 tok/s cleanly.

Not fixed elsewhere either: thecodacus's fork does not contain this cache (its
two optimisations are host pinning and expert prefetch, both excluded here),
upstream PR #26824 is open with only a "freeze swapping during multi-slot
batches" mitigation and no root cause, and the one upstream commit that looked
relevant - `sched: reintroduce less synchronizations during split compute
(#20793)` - is already reverted in our tree (`86b94708f`).

## CORRUPTION — ROOT CAUSE LOCALISED (2026-08-25)

### It IS moe-cache, and it requires ADMISSIONS

Controlled, offload pinned at `-ncmoe 30` so only the cache varies:

| config | VRAM free | simple | trigger |
|---|---|---|---|
| `--moe-cache off` | 4,840 MiB | clean | **clean** |
| `--moe-cache 256` | 4,634 MiB | clean | DEGEN |
| `--moe-cache 512` | 4,326 MiB | clean | DEGEN |

**A 256 MiB cache with 4.6 GB free still corrupts, so it is NOT memory
pressure.** Upstream master with the same `-ncmoe 30` is clean (7,045 MiB used,
4,856 free), and our fork with the cache off matches it exactly (7,061 / 4,840,
clean). The fault is the cache code path at any size.

**It requires admissions**, which is the tightest handle yet:

```
cache 512, defaults            simple=clean  trigger=DEGEN
cache 512, SPEC_EVICT=off      simple=clean  trigger=DEGEN
cache 512, ADMIT_AFTER=255     simple=clean  trigger=clean   <- no admissions
cache 512, INSERTS=1           simple=clean  trigger=DEGEN
```

`ADMIT_AFTER=255` effectively forbids admission and is CLEAN. So corruption
happens when experts are WRITTEN INTO the cache, not when they are read - which
is why every read-side verifier (slot contents, slot->expert mapping, row
accounting, batched-vs-single kernel, dispatch-time re-check) came back clean.
Admissions into a small pool also force evictions, so eviction racing with
in-flight use is the leading remaining candidate.

**The corruption needs ONLY the host->device copy.** With
`GGML_CUDA_MOE_CACHE_FAIL=dispatch` the cache never runs a kernel, and it still
corrupts; with `FAIL=insert` (never admits) it is clean. So the sole remaining
GPU action - an H2D `cudaMemcpyAsync` into the slab, followed by
`cudaStreamSynchronize` - is sufficient to corrupt model weights.

Also excluded under a VALID control (`--moe-cache N`, not the env var):
- evictions: `--moe-cache 4096` corrupts with **evictions = 0**
- weight repacking: `--no-repack` does not help, and explicit modes already
  disable it
- host pinning: `HOSTREG_MB=0` does not help
- speculative eviction: `SPEC_EVICT_MODE=off` does not help
- slab sizing: `n_slots` is set from the FINAL `slot_count` after the
  allocation retry loop, so the bounds check matches the allocation
- fill/dispatch race: the fill does `cudaStreamSynchronize` before marking the
  slot valid, and `serial_fill` defaults true with a single worker

**CUDA graphs: properly rejected at last (2026-08-25).** Rejected twice before
on fake controls; re-run with `--moe-cache off` as the control:

```
cache 512, graphs ON (default) : train=DEGEN  halting=DEGEN
cache 512, GRAPHS DISABLED     : train=DEGEN  halting=DEGEN
cache OFF, graphs ON (CONTROL) : train=clean  halting=clean
```

**The "trigger prompt" was never special.** At 512 MiB the halting prompt
degenerates too. A smaller cache churns admissions faster, so corruption
arrives sooner; larger caches merely delay it. Any framing that treats one
prompt as the trigger is wrong - what matters is how many admissions have
happened.

**THE CACHE IS CORRECT IN `llama-perplexity` - the fault is in how the SERVER
drives it.** Same binary, same cache, heavily exercised:

```
-c 512,  4-token batches : PPL 7.3347 (vs 7.2856 cache off)  1.19M hits, 379k evictions
-c 2048, 4-token batches : PPL 5.0367                        2.28M hits, 780k evictions
-c 512,  1-token batches : PPL 6.7508                        371k hits,   83k evictions
```

Millions of hits and hundreds of thousands of evictions with normal perplexity
(corruption reads as 141+). So fills, hits, evictions, single-token batches and
larger contexts are all sound. Every remaining hypothesis must explain why only
`llama-server` breaks.

**Pin discipline audit: clean.** `GGML_CUDA_MOE_CACHE_PIN_AUDIT=1` flags any
`slot_reset` or refill (`state -> copying`) performed on a slot with
`readers > 0`. Motivated by `moe_cache_slot_reset` zeroing `slot.readers`
UNCONDITIONALLY, which would silently discard a live pin. **Zero violations**
through a corrupting generation, so no slot is ever torn down or refilled while
referenced.

**Fill logging + VRAM canary: the copy is provably correct.**
`GGML_CUDA_MOE_CACHE_LOG_FILLS=1` logs every fill's slab range, destination and
size; `GGML_CUDA_MOE_CACHE_CANARY=1` allocates a 1 MiB 0xA5-filled buffer right
after each slab and re-checks it after every completed fill.

```
CANARY armed: slab=0x7f8e05600000..0x7f8e1f010000 (729 slots x 589824 B)
FILL #0 pool=0 slot=0/729 dst=0x7f8e05600000 dst_end=0x7f8e05690000
        bytes=589824 stride=589824 in-bounds
out-of-bounds fills: 0        canary corruptions: 0        trigger: DEGEN
```

Every fill is in-bounds, the canary is untouched, and the output still
corrupts. So the H2D copy writes exactly where it should, with the right size,
and does not overrun. **Memory overrun is excluded.**

Combined with `FAIL=dispatch` still corrupting (the cache never runs a kernel,
never serves a row), the remaining surface is very small: pin/unpin bookkeeping,
the slot state machine, and whatever the CPU dispatch path does differently
when `moe_cache_node` is non-NULL - notably the dispatch-failure restore of hit
rows into `matrix_rows`.

Checked and NOT the cause: the fill copy overruns its slot. `job.bytes` is
carried from the node while the destination stride is `pool->expert_size`, and
nothing validated they agree - a real missing check, now added and logging
loudly - but it never fires, so the copy is in bounds.

### WHY THIS INVESTIGATION TOOK SO LONG - a methodology failure

`--moe-cache auto` on the command line calls `setenv(..., 1)` with OVERWRITE,
and also `unset_env_var("GGML_CUDA_MOE_CACHE_BUDGET_MB")`:

```
GGML_CUDA_MOE_CACHE=0  PLUS  --moe-cache auto   ->  VRAM 11,743 MiB (cache ON)
```

Every "cache off" control that also passed `--moe-cache auto` was silently
running WITH the cache. That invalidated the graph test, the spec-eviction
test, and made the controlled matrix look like it had "drifted" hours later. It
had not - the controls were fake. It also explains why `BUDGET_MB` appeared
broken: the flag wipes it. The correct API is `--moe-cache N`.

**Rule going forward: verify the control is actually a control.** Assert the
expected VRAM footprint or `slots_used` before trusting any arm.

### Hypotheses tested and rejected along the way

atlas warming; moe-cache wholesale (wrongly retracted, see above); OOM/VRAM
pressure; scratch-capacity overrun; multi-activation mmv mapping; CUDA-graph
capture collision (rejected twice - once invalidly, once with a valid control);
speculative eviction; fill-copy overrun. Also confirmed model-independent:
gemma-4-26B degenerates identically, emitting `<unused49>` where Ornith emits
`/` - both are just whatever token wins on degenerate logits.

### Known-good configuration

`--moe-cache off` - verified clean on the reproducer, 44 t/s, 3.5 GB headroom.
Costs ~21% throughput versus a working cache.

## SUPERSEDED - earlier narrative: moe-cache corrupts output — PRE-EXISTING, still open (2026-08-25)

**Deterministic reproducer**: `--moe-cache auto`, any context size, prompt
"Work through this step by step: a train leaves A at 60mph, another leaves B
200 miles away at 40mph one hour later. Explain your full reasoning." at
max_tokens=600. Output is `////` from character 0.

**Once triggered the server is permanently poisoned** — a subsequent "Name
three primary colours" also returns `////`. Only a restart clears it.

**Pre-existing, not introduced this session.** Reproduces identically on
`5de4ff917` (fingerprint b10601), the build from before this session's work.

What has been ruled out:
- **Not atlas warming.** An earlier isolation blamed warming, but that
  experiment varied max_tokens between arms (500 clean / 600 degenerate) and
  was worthless. With length held fixed, WARM=0 and WARM=1 both corrupt.
- **Not generation length alone** — 800 tokens is clean on a server that has
  not yet seen the trigger prompt.
- **Not any optional subsystem**: HOST_MB=0, HOSTREG_MB=0, COLD_AFTER_S=0,
  GGML_SCHED_PREFETCH_EXPERTS=0, ATLAS_WARM_ASYNC=0/1 all still corrupt.
- **Not offload depth** — corrupts at 27, 28 and 30 CPU MoE layers alike.
- **Not eviction** — `evictions: 0` at the moment of corruption.
- **Not corrupted slot contents.** A sampled verifier
  (`GGML_CUDA_MOE_CACHE_VERIFY_SLOTS=N`) comparing each hit's slot against the
  host weights it mirrors, across head/middle/tail windows, reports **zero
  mismatches** through a full corrupting generation. The cache holds the RIGHT
  DATA. `fill_failures` is also 0.

So it is an **indexing/routing bug, not a data bug**: correct expert weights
reaching the wrong row, or the wrong slot being selected for a row. The row
bookkeeping in `ggml_compute_forward_mul_mat_id` is the place to look; its six
parallel arrays are correctly sized (all MOE_CACHE_MAX_TOPK=4096, matching the
`n_ids * ids->ne[1] <= 4096` gate), and the collect-failure fallback's apparent
missing `matrix_row_counts` increment is NOT a bug — it recomputes each row
immediately after writing the mapping, so reusing the scratch index is safe.

**INVESTIGATION STATUS (2026-08-25, unresolved).** Three separate "found it"
claims were made and all three were WRONG, each because more than one variable
moved between arms:

1. *Atlas warming* - arms differed in max_tokens (500 clean / 600 degenerate).
2. *moe-cache itself* - `GGML_CUDA_MOE_CACHE=0` looked clean, but that arm also
   had `-c 65536`; with `-c 2048 -ncmoe 40` and no cache it degenerates too.
3. *VRAM exhaustion / OOM* - `GGML_CUDA_MOE_CACHE_RESERVE_MB=2560` leaves
   2544 MiB free and it still degenerates 3/3. Not memory pressure.

Contradictory observations that no single variable explains:

| config | result |
|---|---|
| `cache=0, -c 65536` (train prompt included) | clean |
| `cache=0, -c 2048 -ncmoe 40` | DEGENERATE |
| `cache=auto, -c 65536` | DEGENERATE |
| `cache=auto, -c 4096`, 800-token generation | clean |
| `cache=auto, -c 4096`, then train prompt | DEGENERATE |

What IS established:
- Deterministic trigger prompt (the train word-problem), and once triggered the
  server stays poisoned for unrelated later requests until restart.
- Pre-existing: reproduces on `5de4ff917` from before this session.
- Slot CONTENTS are correct (`GGML_CUDA_MOE_CACHE_VERIFY_SLOTS`, 0 mismatches).
- Slot->expert MAPPING is correct (`GGML_CUDA_MOE_CACHE_VERIFY_ROWS`, 0
  mismatches).
- Row accounting BALANCES (`MOE_CACHE_ROW_AUDIT`, 0 mismatches) - no dropped rows.
- Persists with `GGML_CUDA_MOE_CACHE_FAIL=dispatch`, i.e. with all expert math
  on the CPU.
- No `cudaMalloc failed` / OOM lines in any log.
- The cache expands to fill VRAM down to the ~128 MiB reserve floor regardless
  of `-ngl`, which is worth fixing on its own merits but is NOT this bug.

**CONTROLLED MATRIX RUN (36 fresh servers, one per sample so a poisoned server
never contaminates the next cell, max_tokens/temperature/-ngl pinned):**

| cache | ctx | prompt | degenerate |
|---|---|---|---|
| auto | 2048 / 4096 / 65536 | train | **3/3 each** |
| auto | any | halting | 0/9 |
| off | any | either | 0/18 |

`by cache: auto=9/18, off=0/18` · `by prompt: train=9/18, halting=0/18` ·
`by context: 3/12 at every size - NO effect`

So it IS moe-cache, interacting with a specific prompt, deterministic in both
directions. The earlier retraction of that diagnosis was wrong: the arm that
seemed to exonerate the cache (`cache=0 -c 2048 -ncmoe 40`) was broken by a
hand-set `-ncmoe 40` on a model whose expert map has only 30 rows - a
misconfiguration introduced by the investigator, not the cache.

**Greedy (temperature=0) divergence: character 0.** Cache-on and cache-off
diverge on the very first generated token, so the damage is already present
before any decode step.

**Sharp bisect on batch size:**

```
GGML_CUDA_MOE_CACHE_MAX_BATCH=1  -> clean
                              2  -> DEGENERATE
                              4/8/16/64 -> DEGENERATE
```

`max_batch=1` admits only single-token nodes. Everything >= 2 corrupts. This is
the tightest handle on the bug so far.

**Hypothesis tested and REJECTED**: that the multi-activation path is at fault.
`ggml_cuda_moe_cache_mmv`'s parameter mapping carries the comment "ne12 = ne13
= 1 (single token)", suggesting it is only correct when all hits share one
activation row. Refusing nodes with `!shared_activation` did NOT fix the
corruption, so `shared_activation` is true in the failing case and
multi-activation is not the mechanism. Guard reverted rather than left in as a
false fix.

**Hypothesis SIX: CUDA-graph capture collision - REJECTED (2026-08-25).**
The cache's fill worker issues cudaMemcpyAsync on its own stream while a graph
capture may be live on the main stream, and plan.md already records this codebase
hitting exactly that ("operation not permitted on an event last recorded in a
capturing stream ... Gemma's capture boundaries evidently avoid this;
qwen35moe's do not"). Ornith is qwen35moe-class. It also uniquely explained the
shape of the evidence: a timing collision corrupts nothing that a state check
would see, which is why every verifier comes back clean. Measured:

```
graphs ON  (default), cache on : DEGEN DEGEN DEGEN
graphs OFF, cache on           : DEGEN DEGEN DEGEN
```

```
graphs OFF, cache off (CONTROL) : DEGEN DEGEN DEGEN
```

**THE CONTROL ALSO DEGENERATED, so this test proves nothing** - not the
hypothesis, not its rejection. Cache-off was clean 18/18 in the controlled
matrix earlier the same day and fails 3/3 here, so the machine or build state
has drifted between the two and the graph arms cannot be interpreted.

The hypothesis is therefore UNTESTED, not rejected. Re-running it requires
first re-establishing that cache-off is clean on the current build - i.e.
re-validating the control before trusting any arm.

That drift is itself the most important open fact: **the reproducer's
dependence on cache state is no longer stable**, which undermines the
controlled matrix's central finding (auto 9/18 vs off 0/18). Everything
downstream of that matrix needs re-confirmation.

**Excluded so far** (all re-verified under a VALID control - `--moe-cache off`
on the CLI - with a conservative detector that requires unique-char ratio <0.02
or a 20+ character run): atlas warming; OOM / VRAM pressure (256 MiB cache with
4.6 GB free still corrupts); scratch overrun; multi-activation mmv mapping;
CUDA graphs; speculative eviction; fill-copy overrun (canary + per-fill logging
clean); evictions (corrupts with evictions=0); weight repacking; host pinning;
admission demand thresholds; cold-page sweep; group admission; prediction
measurement; slot count and continuous batching; KV unification and prefix
reuse; context size; pin discipline (audit clean); the copy itself (SKIP_COPY
still corrupts); and row routing (NO_HITS still corrupts).

**Required**: admissions. `FAIL=insert` is the only cache-enabled configuration
that is clean.

### 2026-08-25: two more hypotheses built, instrumented and FALSIFIED

**Runtime allocation in the fill worker** (`GGML_CUDA_MOE_CACHE_NO_RT_ALLOC=1`).
The worker lazily ran `cudaStreamCreate` / `cudaMallocHost` / `cudaFreeHost` -
all device-wide synchronizing - on a background thread *during decode*,
concurrently with the main thread's capture and dispatch. It was the only
channel reached by every corrupting arm and short-circuited by `FAIL=insert`,
the sole clean one. The flag preallocates stream and a 64 MiB pinned stage at
worker startup and never allocates again. Verified active
(`[moe-cache] NO_RT_ALLOC stream=preallocated stage=64 MiB`). **Still DEGEN**,
byte-identical to control (uniq=0.0017).

**The CPU-side host_ptr redirect** (`GGML_CUDA_MOE_CACHE_NO_HOST_PTR=1`).
`ggml_compute_forward_mul_mat_id` redirects CPU expert reads to a cache-owned
host copy. This is the one cache-touched path that SKIP_COPY (device-only),
FAIL=dispatch (GPU kernel) and NO_HITS (GPU row routing) all leave running, and
it is gated on admission. **Still DEGEN.**

**Ruled out by reading, not testing**: the persisted selection history
(`GGML_CUDA_MOE_CACHE_HISTORY` is opt-in, unset, no file on disk - so it is not
a cross-restart carrier); host promotion's `MADV_DONTNEED` on source pages
(`moe_cache_host_budget_bytes()` returns 0 by default, so promotion never runs).

**Method fix.** Arms now probe `simple` BEFORE the trigger. Every arm's
`simple-BEFORE` is clean, which establishes that each fresh server is born
clean and that **the poisoning does not survive a process restart** - only
within a server's lifetime. Earlier A/B arms were therefore not contaminated by
a preceding degenerate arm, but the check is now permanent.

**The sharpened contradiction.** Admissions are necessary, yet *no consumer of
the cached data matters*: not the device copy, not GPU dispatch, not GPU row
routing, not the CPU redirect, not runtime allocation. Whatever breaks is in
the bookkeeping or state machine that admission drives, not in the bytes it
moves or in anything that reads them.

**Remaining surface**: whatever differs between `llama-server` and
`llama-perplexity` in how they drive `llama_decode` around an admitting cache. (atlas warming, moe-cache
as a whole, OOM/VRAM, scratch-capacity overrun, multi-activation mapping). The
bug is real, deterministic, and still unexplained.

**Current stopgap**: `GGML_CUDA_MOE_CACHE_MAX_BATCH=1` keeps the cache ON and
active (hit rate 0.70, 5650 slots) and is clean 4/4 including the reproducer,
at 44 t/s - the same as cache-off, so the multi-token path was carrying the
throughput benefit. Better than disabling the cache outright.

**Next step**: instrument inside the `max_batch >= 2` path specifically -
compare a multi-token node's GPU result against a CPU recomputation of the same
rows, which turns "output is wrong" into "row R of node N differs", the fact
that will finally name it.

**Superseded note - the original next step was a controlled matrix** - cache x context x prompt, one
variable at a time, repeated - not another hypothesis. Related prior art: a
dangling commit (`c45ae5b309`, 2026-08-13) documents a real OOM-vs-throughput
interaction with `cudaMalloc failed: out of memory` on the KV cache, and the
user recalls an uncommitted "loud failure" guard for this class of problem
that has not been located (dangling candidates are either already merged - the
64->4096 ceiling - or superseded).

**No configuration is currently known-safe.** `GGML_CUDA_MOE_CACHE=0`, which costs ~21%
throughput (44 t/s vs ~56).

### Consequence for this session's server-measured numbers

Degenerate output generates FASTER than clean (68-69 t/s vs 55-56), so any A/B
arm that silently corrupted would read as a throughput win. Every server
measurement taken before this was found lacked a degeneration check.

Re-measured with every sample degeneration-checked (0/4 degenerate in all
arms), 2 reps:

| arm | tok/s |
|---|---|
| substitution OFF | 57.81, 57.67 |
| substitution ON | 60.55, 60.34 |

**+4.7%, confirmed real** (previously reported as +5.4% unguarded). The
perplexity bound is unaffected — those runs used `llama-perplexity`, where
corruption would have produced absurd PPL rather than 6.31.

Still to re-measure with the guard: the admission A/B (-7.6% for removing the
gate), the atlas warming A/B (+0.27 tok/s), and the fill-cost model calibrated
from those arms.

## Fold thesis — VALIDATED: real 3-way structure beyond pairs (2026-08-25)

Groups of experts fire together as sets, not merely as correlated pairs. Tested
on 113,685 pooled decisions, held-out split, triples with count >= 5, scored
against the Kirkwood superposition (what the three pairwise marginals predict):

```
mean log-lift  +1.050    (2.9x more often than pairwise predicts)
median         +1.017
share above pairwise prediction:  96.7%
strongly above (>+0.5):           83.1%
```

**96.7% of triples exceed what pairwise structure explains.** Every mechanism
built so far - co-activation substitution, the atlas, all the eviction arms -
reads only the pairwise shadow of this.

Order limits: recurrence falls with order (2-way 65.3% of train groups recur
held-out, 3-way 30.4%, 4-way 14.4%) and seen-once rises (30.4% / 54.7% /
70.1%). Triples are the usable order; 4-way is mostly noise at this sample size.

Note the selection process constrains exactly 8 of 256 per decision, which
induces *competition* between experts and would bias higher-order correlation
NEGATIVE - so a +1.05 positive lift is not a selection artifact.

**Actionable**: substitution currently picks a stand-in by pairwise
co-activation. Selecting by triple co-occurrence with the other experts in the
same decision is the direct test of whether the higher-order structure converts
into a better mechanism, and it reuses data already collected.

## Commit audit — behaviour changes shipped in the days before the bug (2026-08-25)

The corruption reproduces on `5de4ff917` (08-24 09:33), so it predates this
session. Auditing what landed on the MoE execution path in the days before,
most commits are instrumentation, measurement or reverts. The ones that changed
DEFAULT behaviour for every user:

- **`107389751` (08-23 20:04) "default SPEC_EVICT_MODE to agree, not off"** -
  ships SPECULATIVE EVICTION on by default. Its own message argues "Safe even
  though it's now the default". Evicting a slot on a *prediction* is precisely
  the shape of a bug that corrupts output while every state check passes, and
  it landed before the earliest commit the corruption is confirmed on.
  **Leading suspect.** Disable with
  `GGML_CUDA_MOE_CACHE_SPEC_EVICT_MODE=off`.
- `ef2c30b43` (08-23 19:56) - fixes agree-mode at depth 3+, same subsystem.
- `fce3e9fa5` (08-23 19:30) - introduces cross-depth-agreement spec eviction.
- `30d0f8447` (08-23 20:25) - mincore-verified NVMe cost tier (read-only, low risk).
- `b321b6259` (08-24 04:27) - prefill copy-failure guard (a FIX for a prior
  silent-corruption bug in the same area).

Test queued: control (cache off) FIRST - the previous graph test was void
because its control degenerated - then spec_evict default vs off.

## DESIGN ARGUMENT — read this before proposing new cache work (2026-08-25)

The reasoning behind the current direction, recorded because several sessions
have re-derived (and mis-derived) it. The measurements backing each claim are
in the sections below.

### 1. Hit rate is the wrong metric for this design

Hit rate is what a FETCH-based cache lives or dies by: a miss means a stall, so
minimising misses is the whole game. Under SUBSTITUTION a miss costs no time at
all - it costs a slightly different expert. The right measure is **retained gate
mass**: how much of the weight the router intended actually got executed.

The two diverge violently at low residency. From 18.5% down to 1.2%:

- hit rate collapses 96.9% -> 38.7% (looks fatal)
- retained gate mass degrades 99.1% -> 80.4% (entirely workable)

An earlier assessment in this project extrapolated from hit rate and concluded
the GLM-scale vision was out of reach. That conclusion was WRONG, and wrong
specifically because it used the fetch-design metric on a substitution design.

Hit rate is also *purchasable*, which disqualifies it as an objective on its
own: removing the admission gate buys +10.79pp of hit rate and costs 7.6% of
throughput. Any optimisation target has to price fills.

### 2. Why substitutes stay good as the cache shrinks (structural, not luck)

Stand-in quality is FLAT at ~230% of the replaced pick's router score from 5%
residency down to 0.6%. The substitute is consistently rated HIGHER than what
it replaced. That is not an artifact:

- Misses concentrate on the router's weak picks - rank 7 misses 44% of the time
  against rank 0's 20%, monotonic across all 8 ranks.
- So the expert that goes missing is usually one the router rated poorly.
- Whatever IS resident is, by construction, the frequently-selected hot core -
  experts the router rates highly.
- Shrinking the cache concentrates it further onto that core, which keeps the
  relationship intact rather than eroding it.

This is why the design should be expected to scale to models where residency is
1% rather than 18%.

### 3. The binding constraint at scale is COVERAGE, not accuracy

The number that actually degrades with residency is the **no-candidate tail** -
picks with neither a hit nor any resident stand-in, which fall back to CPU:

```
18.5% residency ->  0.68%
 5.0%           ->  2.56%
 1.2%           -> 15.62%
 0.6%           -> 22.86%
```

At GLM-scale residency roughly one pick in six has nothing usable resident.
That, not stand-in quality, is what would set throughput. It is a different
problem from everything worked on so far: it asks "is SOMETHING usable always
reachable?" rather than "is the RIGHT expert present?".

### 4. Consequence — an eviction objective nobody has tried

Six signals have been tested as eviction policies and all lost to plain LRU
(atlas, router score, co-activation, frequency, admission demand,
anti-co-firing). But every one of them was scored on **hit rate or retained
mass** - i.e. on getting the right expert resident.

Nobody has asked a policy to maximise **coverage**: keep at least one viable
stand-in reachable for every likely pick, accepting a worse hit rate to do it.
That is a group property rather than a per-expert one, so it is also the most
plausible place for the validated 3-way fold structure (96.7% of triples exceed
pairwise prediction) to finally earn its keep - a policy that keeps one member
of each active group resident, rather than the k hottest individuals.

### 5. What is settled and what is open

**Settled - stop re-testing these:**
- Prefetching cannot pay. Capped at +0.30pp even with a perfect topic oracle,
  because every topic touches ~6,000 of 7,680 experts.
- Placement/"which experts to keep" is effectively solved. LRU wins, and Belady
  bounds ANY policy's remaining headroom at ~17pp on hit rate.
- The atlas is neutral-to-marginal as a policy input and cannot currently be
  evaluated through the server at all (warming touches ~2% of fills).
- Filling harder loses. Removing admission control costs 7.6% throughput.

**Open, in priority order:**
1. The corruption bug - blocks every live measurement.
2. Perplexity at LOW residency. The +0.47% bound is an 18.5% number;
   substitution actually bites at 1.2%. This does NOT need the bug fixed -
   `llama-perplexity` with a small `GGML_CUDA_MOE_CACHE_BUDGET_MB` measures it.
   It is the single result that could still sink the design.
3. Coverage-oriented eviction (section 4).
4. Scale validation: 4,713 units here, low cells only 28-117 slots. Whether a
   750B model's hot core stays this compact is untested.

## Embedding implementation fixed — and why probes beat co-occurrence (2026-08-25)

The learned embedding scored AUC 0.4900 on held-out link prediction while a
trivial common-neighbours count scored 0.6530. That was an implementation
failure, now corrected by dropping SGNS and representing each expert directly
by its row of co-firing counts:

| representation | AUC |
|---|---|
| co-occurrence rows, cosine (**the fix**) | **0.6345** |
| + truncated SVD to 16 / 32 / 64 dims | 0.6072 / 0.6126 / 0.6212 |
| + PPMI weighting | 0.5996 |
| common neighbours (trivial baseline) | 0.6530 |
| previous SGNS embedding | 0.4900 |
| **probe atlas (9 categories)** | **0.7116** |

Two findings beyond the fix:

**Dimensionality reduction actively hurts.** SVD to 64 dims loses ground and
PPMI - the textbook weighting for co-occurrence - is worse still. The signal is
in the full high-dimensional pattern; there is no low-dimensional manifold to
recover. This is why the spectral discovered atlas was never going to work.

**Yet 9 probe dimensions beat 256 co-occurrence dimensions.** So the problem is
not dimensionality - it is what the dimensions measure. Probes measure a
FUNCTIONAL property (what causes this expert to activate); co-occurrence
measures a CORRELATIONAL one (what happens to fire alongside it). For
predicting FUTURE co-firing the functional property generalises and the
correlational one memorises.

That now explains every discovered-atlas result in this document: spectral
(0.5761), incremental SGNS (0.4823-0.4900), and raw co-occurrence rows (0.6345)
have all lost to the probe atlas (0.7116). Four independent methods, one
verdict - and the verdict is about the SIGNAL, not the method. Keeping the
hand-written probes is the better-supported position.

## Low-residency perplexity — partial result (2026-08-25)

Substitution's quality cost measured with `llama-perplexity`, 60 chunks per
arm, paired per-chunk against a substitution-OFF baseline, residency varied via
`GGML_CUDA_MOE_CACHE_RESERVE_MB` (`BUDGET_MB` is ignored - verified: identical
11,797 MiB peak at 3413 and 256):

| arm | delta PPL | 95% CI | significant | worse chunks |
|---|---|---|---|---|
| full residency | -0.217% | [-0.597%, +0.165%] | no | 28/60 |
| RESERVE 1500 | -0.035% | [-0.462%, +0.395%] | no | 31/60 |
| **RESERVE 2500** | **+0.998%** | **[+0.148%, +1.855%]** | **YES** | 36/60 |
| RESERVE 3300 | -0.069% | [-0.433%, +0.296%] | no | 29/60 |

**At full residency substitution is free** - CI spans zero, 28/60 chunks worse,
a coin flip. Independently confirms the earlier 120-chunk result (+0.181%, CI
[-0.106%, +0.468%]) on a different corpus slice.

**At one reduced-residency point it costs ~1%** - the first statistically
significant quality cost measured for substitution anywhere in this project,
and still small.

**The curve is NOT established.** The ladder was not clean: RESERVE 3300 shows
ZERO moe-cache log lines, i.e. the cache never allocated at all, so there were
no substitutions and its null result is meaningless as a low-residency point.
That leaves two live rungs and no way to tell their residency apart, because
`llama-perplexity` exposes no `slots_used`.

Methodological note: RESERVE_MB was used as a residency proxy without verifying
it produced a graded ladder. It does not - it goes full -> reduced -> off, with
no readback. **To finish this measurement the cache needs to report residency
from a non-server binary** (a stderr summary at exit would do), or the server
route, which is blocked by the corruption bug.

## The six eviction signals RETESTED — two of the verdicts were wrong (2026-08-25)

The six signals were each tested as a WHOLE policy at 18.5% residency. The
coverage result proved both choices can hide a real signal (redundancy: -18pp
as a whole policy, +4.2pp as a tie-break, and only at low residency). Retested
in the tie-break shape (LRU narrows a 32-sample to its M=8 stalest, the signal
breaks that tie) at 1.2% residency, scored on retained gate mass:

| signal | hit | no candidate | retained | vs pure LRU |
|---|---|---|---|---|
| (pure LRU) | 39.70% | 17.69% | 78.30% | - |
| **coverage** | 36.89% | 14.48% | 82.48% | **+4.18pp** |
| **atlas** | 39.39% | 16.61% | 79.46% | **+1.17pp** |
| **router score** | 36.20% | 17.78% | 79.01% | **+0.71pp** |
| co-activation | 39.79% | 19.40% | 76.24% | -2.06pp |
| frequency | 38.25% | 20.04% | 75.58% | -2.72pp |

**The atlas and router score verdicts were wrong.** Both were reported as
failures - the atlas as losing even to random - and both are positive in the
correct shape and regime. Co-activation and frequency remain negative, so those
two verdicts stand.

The methodological lesson, now demonstrated three times (anti-co-firing,
coverage, and these two): **a signal's value depends on the shape it is used in
and the regime it is measured at.** Testing a signal as a whole policy at high
residency is close to a worst case for it - recency dominates when the cache is
large, and any signal given full control fights it. Future signal tests must
report shape and residency alongside the result.

Untested and now the obvious next step: **combinations** - coverage as primary
tie-break with atlas or router score as a secondary, since the three positive
signals may be capturing different structure.

## Combined eviction signals — all three, softly (2026-08-25)

Once retested in the tie-break shape, three signals are positive (coverage
+4.18pp, atlas +1.20pp, router score +0.70pp). Combined at 1.2% residency, LRU
narrowing a 32-sample to its M=8 stalest:

| combination | hit | no candidate | retained | vs LRU |
|---|---|---|---|---|
| coverage | 36.89% | 14.48% | 82.48% | +4.18pp |
| coverage+atlas | 38.54% | 13.57% | 83.31% | +5.01pp |
| coverage+score | 36.80% | 14.42% | 83.17% | +4.87pp |
| atlas+score | 38.08% | 14.78% | 82.64% | +4.34pp |
| **all three (rank-sum)** | 37.56% | **13.35%** | **84.23%** | **+5.93pp** |

Every pair beats its members and the triple beats every pair, so the three are
capturing different structure rather than restating one signal.

### A hierarchy was tried and LOST

Reasoning said coverage should be a hard VETO (never evict the sole resident
representative of a neighbourhood), then atlas for relevance, then score for
value. Measured:

| policy | retained | vs LRU |
|---|---|---|
| rank-sum of all three | 84.23% | +5.93pp |
| coverage veto -> LRU | 78.31% | +0.02pp |
| coverage veto -> atlas -> score | 78.82% | +0.52pp |
| coverage veto -> score -> atlas | 79.30% | +1.01pp |

**The veto is worth nothing.** It only excludes redundancy == 0, and at 1.2%
residency almost nothing has zero redundancy, so it rarely fires and the policy
degenerates to LRU.

**Redundancy's value is in its GRADIENT, not its extreme.** "How redundant"
carries signal across its whole range; "is it the very last one" is a threshold
that almost never trips. Promoting a graded signal to a hard rule destroys the
part doing the work - the same failure mode as testing signals as whole
policies, one level down.

Untested refinement: WEIGHTED rank-sum, with coverage weighted above atlas and
score to reflect their individual strengths, which keeps every gradient while
respecting that they are not equally informative.

## Gemma UD-quant substitution test — INVALID, baseline broken (2026-08-25)

Question: does substitution cost more on an Unsloth dynamic quant, whose bit
budget is spent on attention and keeps MoE experts thinnest? That matters for
the 2-bit GLM plan, where quantisation and substitution would degrade the same
component.

Ran gemma-4-26B-A4B-it-UD-Q4_K_M, same corpus and settings as Ornith:

```
off  PPL = 141.10 +/- 7.06
on   PPL = 139.65 +/- 6.98
paired -1.026%  CI [-4.434%, +2.502%]  not significant, 25/60 chunks worse
```

**The baseline is broken.** PPL 141 against Ornith's 4.34-7.04 on the identical
corpus is not a quantisation difference - the model is producing near-garbage
regardless of substitution, so on/off against it means nothing. The +-4.4% CI
also swamps the ~1% effects being hunted.

No warnings in the log and the config loaded normally (n_ctx=512, batch 4).
Most likely cause: gemma-4 uses interleaved local/global attention with a
sliding window, and n_ctx=512 is too small for it on raw technical prose.
Ornith tolerates that context size; gemma evidently does not.

**To answer the question properly** the gemma arm needs a config where its
baseline PPL is sane (larger n_ctx, and possibly a corpus closer to its
training distribution), verified BEFORE comparing substitution on/off. Until
then the UD-quant interaction with substitution is unmeasured, and the 2-bit
GLM plan rests on an untested assumption.

## How to combine eviction signals — four forms tried (2026-08-25)

Criticism that prompted this: sum vs product was being chosen by result rather
than by what the quantities mean. Correct, and worth resolving properly.

The eviction decision should minimise expected future loss:

```
Loss(evict X) = P(X requested soon) x value(X) x P(no good substitute)
```

which is a product of probabilities, i.e. a SUM OF LOG-PROBABILITIES. So
addition and multiplication are the same model in different spaces; what was
genuinely arbitrary was combining raw RANKS, which are neither. Also noted:
recency is itself an estimator of P(requested soon), so treating LRU as a
separate narrowing stage rather than a factor was a modelling error.

Results, all at 1.2% residency:

| form | vs LRU |
|---|---|
| hierarchy (coverage veto -> relevance -> value) | +0.02 to +1.01pp |
| product of raw quantities | +4.72 to +4.84pp |
| calibrated log-additive expected loss | **-9.51pp** |
| **weighted rank-sum (heuristic)** | **+6.18pp** |

**Why the principled forms lost - both times it was the inputs, not the form.**
The product let whichever factor had the widest dynamic range dominate (score
EMA spans orders of magnitude, cosine is bounded in [-1,1], redundancy is a
small integer), silently discarding two of three signals. The log-additive
model was calibrated with a 2000-decision reuse window, which turned out to be
so long that P(reused) sits between 0.86 and 0.99 - almost everything returns,
so the term carries nearly no information and the value term takes over.
Recency's predictive content is not WHETHER an expert returns but HOW SOON.

**Why rank-sum keeps winning**: ranks are invariant to miscalibration. Every
principled form needs its inputs correctly scaled or calibrated, and that
requirement was violated both times. Rank aggregation needs only each signal's
ORDERING to be informative - a much weaker condition, and the one these signals
actually meet.

That is not a claim that rank-sum is theoretically superior. It is the right
tool for three signals of unknown calibration, which is the situation here. A
log-additive model remains the better target IF the factors are calibrated
properly - and the failure above shows the calibration, especially the reuse
window, is where the work is.

## Signal weighting and orthogonality — the eviction recipe (2026-08-25)

A flat 3-way rank-sum lets coverage be outvoted 2:1 by atlas and score, which
both measure an INDIVIDUAL expert's worth, while coverage is the only one
measuring STRUCTURE. Weighting coverage equal to their sum is better:

| weighting (cov, atl, scr) | no candidate | retained | vs LRU |
|---|---|---|---|
| 1, 1, 1 (flat sum) | 13.35% | 84.26% | +5.96pp |
| **1, 0.5, 0.5** (coverage = atlas+score) | **13.06%** | **84.47%** | **+6.18pp** |
| 3, 1, 1 | 13.33% | 84.02% | +5.73pp |
| 4, 1, 1 | 13.73% | 83.51% | +5.21pp |
| 1, 0.5, 0 (drop score) | 13.50% | 83.53% | +5.23pp |
| 1, 0, 0.5 (drop atlas) | 13.47% | 84.07% | +5.77pp |

Over-weighting coverage past parity HURTS, so this is not "coverage matters
most" - the balance between structure and individual worth is what matters.

### The three signals are near-ORTHOGONAL

Spearman rank correlation measured on the actual candidate pools they rank
(n=64,890 pools):

```
cov~atl   -0.0611      cov~scr   +0.1139      atl~scr   -0.0210
```

Coverage and atlas are essentially independent, and no pair exceeds |0.11|.
They measure three different things: **structure** (will the neighbourhood
survive an eviction), **direction** (where is the request heading), and
**realised value** (what has this expert been worth lately).

That orthogonality explains the combination results - every pair beats its
members and the triple beats every pair - and it is why summing recovers more
than any hierarchy. A correction to an earlier inference in this session:
the ablation gap (dropping atlas costs -0.41pp, dropping score -0.95pp) was
read as atlas being redundant with coverage. It is not. At that scale those are
interaction effects, not evidence of overlap.

**Current best recipe**: LRU narrows a 32-candidate sample to its M=8 stalest,
then evict by rank-sum with weights coverage 1.0, atlas 0.5, score 0.5.
+6.18pp retained gate mass at 1.2% residency, no-candidate 17.69% -> 13.06%.
Simulation only; needs a second trace set and more seeds before it justifies
C++ work.

## Coverage-oriented eviction — FIRST policy to beat LRU (2026-08-25)

Predicted by DESIGN ARGUMENT section 4: six signals had failed as eviction
policies, but every one was scored on getting the RIGHT expert resident. This
asks instead for COVERAGE - keep at least one viable stand-in reachable.

Signal: **redundancy** - how many of an expert's co-firing partners are still
resident. Evict the most redundant; protect the sole resident representative of
a neighbourhood.

**As a whole policy it is catastrophic**: no-candidate rises 12-18pp and hit
rate collapses (24% vs LRU's 71% at 5% residency). Redundancy correlates with
being well-connected, which is the hot core, so "evict the most redundant"
evicts precisely the experts that both get hit and serve as stand-ins.

**As a tie-break inside LRU it wins.** LRU narrows a 32-candidate sample to its
M stalest; redundancy picks the most redundant of those. M=1 is pure LRU:

| resident | M | hit | no candidate | retained mass |
|---|---|---|---|---|
| 2.5% | 1 | 54.51% | 8.05% | 89.33% |
| 2.5% | 8 | 51.29% | **6.17%** | **92.01%** (+2.68pp) |
| 1.2% | 1 | 39.70% | 17.69% | 78.29% |
| 1.2% | 8 | 36.90% | **14.44%** | **82.52%** (+4.23pp) |

It LOSES hit rate (-2.8pp) and WINS retained gate mass (+4.23pp) by cutting the
no-candidate tail. That is the trade section 1 of the design argument says is
worth taking, and it is invisible to any hit-rate-scored experiment - which is
why six previous signals missed it.

Two properties that matter:
- **The gain grows as residency falls** (+2.68pp at 2.5%, +4.23pp at 1.2%) -
  the direction needed for the GLM-scale case at ~1.5% residency.
- **Monotone in M** up to 8, so the optimum has not been found; M>8 untested.

Same structural lesson as anti-co-firing: a signal that is destructive as a
whole policy can still be useful as a tie-break inside recency. The difference
is that this one survives the tie-break test and anti-co-firing did not.

Not yet implemented in the cache - this is a simulation result on
`traces/big-*.txt`, and it needs the per-expert resident-partner count
maintained cheaply on the eviction path before it could ship.

## SUBSTITUTION FAILS AT LOW RESIDENCY — measured, and it sinks the design there (2026-08-25)

The measurement repeatedly flagged as the one that could sink substitution.
`llama-perplexity` (where the cache is correct), `--moe-cache 512` = 852 slots,
27% hit rate:

| config | PPL |
|---|---|
| substitution OFF | **6.74** |
| substitution ON | **41.32** |

**A 6x degradation.** At high residency substitution was free (+0.181%, CI
spanning zero, measured twice). At this residency it destroys the model.

This retires the optimistic reading of the cache-size sweep. That sweep showed
retained gate mass degrading gracefully (99% -> 80% at 1.2% residency) and was
used to argue the GLM-scale vision was tractable. **Retained gate mass was a bad
proxy** - it was flagged as a proxy at the time, and it has now been falsified
by direct measurement. Executing 80% of the router's intended weight does NOT
mean the output is 80% as good; substituting the missing 20% with resident
experts wrecks it.

Consequence for the design: substitution is viable ONLY at high residency,
which is precisely where it is least needed. The "8B working set of a 750B
model" case runs at ~1.5% residency, and this says substitution cannot carry
that. The mechanism is not dead - it is free at high residency and still worth
+4.7% throughput there - but it does not scale down, and the scale argument was
the reason to care about it.

## Coverage eviction in the real cache — does NOT reproduce (2026-08-25)

Implemented (`GGML_CUDA_MOE_CACHE_COVERAGE_EVICT=1`): bounded partner adjacency
mirrored from co-activation, and a rank-combined victim choice over the existing
LRU window (coverage 1.0, atlas 0.5, heat 0.5).

It shows the predicted *signature* - it gives up hit rate (0.2711 -> 0.2678
without substitution) - but not the predicted *benefit*:

| eviction | decline rate | hit rate | PPL |
|---|---|---|---|
| LRU | 18.59% | 0.2861 | 41.32 |
| coverage | **19.01%** | 0.2802 | 42.18 |

`substitute_declined` is the real-system counterpart of the simulation's
"no candidate" tail, and coverage eviction made it slightly WORSE, not the
-3.3pp the simulation predicted (+6.18pp retained mass).

Two implementation defects were found and FIXED, and it still does not
reproduce: the partner adjacency was only populated while the flag was on (so
it was empty exactly when the cache first fills), and the third signal was
`weighted_heat` where the simulation used the ROUTER SCORE EMA - a quantity
`plan()` cannot see, since it receives ids and never gate weights. After both
fixes:

| eviction | decline rate | hit rate |
|---|---|---|
| LRU | **18.39%** | 0.2895 |
| coverage+atlas+heat | 18.93% | 0.2824 |
| coverage+atlas only | 18.77% | 0.2791 |

**ROOT CAUSE: the simulation modelled a different eviction architecture.**

```
MOE_CACHE_EVICT_WINDOW = 8
moe_cache_pick_coldest_unpinned(device, pool, pool.lru_head)
lru_head = "probation segment (new/one-off admissions)"
```

The real cache only ever considers the PROBATION segment - 8 new/one-off
admissions. The protected segment (proven-hot experts) is never an eviction
candidate. The simulation sampled 32 slots at random from the ENTIRE resident
set and took the 8 stalest, a pool containing hot, well-connected experts -
and discriminating among those is exactly where the coverage signal earns its
+6.18pp. In the real system every candidate is a fresh one-off admission with
near-identical redundancy, so the ranking has nothing to separate.

The +6.18pp was real for the architecture simulated. Realising it would mean
letting eviction consider the protected segment too, which changes LFRU's
central guarantee and is a much larger design change than a ranking tweak.

**Lesson**: an offline policy simulation must model the candidate SET the real
policy sees, not just the scoring function. Every eviction result in this
document that used the sample-of-32 pool inherits this caveat.

**Protected segment made evictable** (the change the root cause implied), so
candidates come from both segments as the simulation's pool did:

| eviction | decline rate | hit rate | PPL |
|---|---|---|---|
| LRU (probation only) | 18.57% | 0.2878 | 42.66 |
| coverage, both segments | **18.28%** | 0.2661 | 49.24 |
| coverage, both, heat_w=0 | 20.20% | 0.2424 | 51.05 |

The target metric finally moves the right way - decline rate -0.29pp - but that
is a tenth of the simulation's -3.3pp, and it costs 2.2pp of hit rate and
weakens LFRU's guarantee that a proven-hot expert stays resident.

**COVERAGE EVICTION HAS NO REGIME WHERE IT PAYS.** Its entire purpose is to
keep a viable stand-in reachable, i.e. to serve SUBSTITUTION. But:

- at LOW residency, substitution itself is broken (PPL 6.74 -> 41.32), so
  optimising its decline rate optimises a mechanism that must not be used;
- at HIGH residency, substitution is free but the no-candidate tail is already
  negligible (0.68% at 18.5% residency), so there is nothing left to win.

The objective is sound and the implementation now reproduces its direction, but
the window where it would matter does not exist. Kept behind
`GGML_CUDA_MOE_CACHE_COVERAGE_EVICT=1`, off by default, as a working
implementation should the substitution picture change.

## Cache-size sweep — stand-in quality holds as residency collapses (2026-08-25)

The question behind the GLM-scale vision: extract a small working set of a huge
model into VRAM and run the rest from host RAM. Simulated with the SHIPPED
logic (LRU eviction, demand admission, rank-ordered stand-in selection), 113,685
decisions, 4,713 distinct (layer, expert) units:

| resident | slots | hit | no candidate | stand-in vs missed | retained mass |
|---|---|---|---|---|---|
| 18.5% | 871 | 96.90% | 0.68% | 277.1% | 99.11% |
| 10.0% | 471 | 87.21% | 0.79% | 252.7% | 98.93% |
| 5.0% | 235 | 70.31% | 2.56% | 232.0% | 96.33% |
| 2.5% | 117 | 53.66% | 7.62% | 225.1% | 89.55% |
| **1.2%** | 56 | 38.67% | 15.62% | **227.6%** | **80.37%** |
| 0.6% | 28 | 26.80% | 22.86% | 231.5% | 73.11% |

**Stand-in quality does not degrade.** 277% -> ~230% and then flat from 5% down
to 0.6%: even with a 0.6% working set the substitute is still rated HIGHER by
the router than the pick it replaced. Structural reason: misses concentrate on
the router's weak picks (rank 7 misses 44% against rank 0's 20%), so whatever
is resident tends to outrank what was lost, and shrinking the cache only
concentrates it further onto the hot core.

**This is why hit rate is the wrong metric for this design.** Hit rate collapses
96.9% -> 26.8%, but retained gate mass degrades gracefully 99% -> 73%. At the
1.2% residency corresponding to an "8B working set" of a ~750B model, a
fetch-based design faces a 61% miss rate and is bandwidth-dead, while
substitution executes **80% of the router's intended weight** with only 15.6%
of picks falling back to CPU.

Caveats:
- Retained gate mass is a PROXY. The +0.47% perplexity bound was measured at
  18.5% residency; nobody has measured quality at 1.2%, which is where
  substitution actually bites.
- The no-candidate tail grows fast (0.7% -> 15.6% -> 22.9%). At very low
  residency the cost shifts from PCIe bandwidth to CPU compute.
- Scale is NOT proven: 4,713 units here, and the low cells are 28-117 slots. A
  750B model is far wider and whether its hot core stays this compact is
  untested.

## Substitution stand-in selection — router score beats co-activation by 78%

The substitution mechanism picks a stand-in by pairwise co-activation, but that
selector was never validated - only substitution on/off was measured. Tested
offline on 151,584 decisions, scoring each method by the ROUTER'S OWN score for
the stand-in it chose (a good substitute is one the router also rated highly
for this token):

| method | mean stand-in score | vs missed expert | vs random |
|---|---|---|---|
| random resident | 0.01387 | 87.9% | - |
| pairwise co-activation | 0.02132 | 130.9% | **+53.7%** |
| triple co-occurrence | 0.02175 | 133.9% | +56.9% |
| **router score among resident** | **0.03792** | **238.2%** | **+173.5%** |

Three conclusions:

1. **The pairwise selector is real work**, +53.7% over random. It was doing
   something, contrary to doubt.
2. **Triples add only +2.0% over pairs** for this decision. The 3-way structure
   is genuinely there (96.7% of triples exceed pairwise prediction) but does
   not convert into better stand-in selection.
3. **Router score is the right selector** - 78% better than the current
   pairwise implementation. Co-activation is a *proxy* for "which expert suits
   this input"; the router computed that answer exactly, and selecting by
   co-activation discards it.

Stand-ins scoring ABOVE 100% of the missed expert is expected, not anomalous:
misses concentrate on the router's weak picks (rank 7 misses 44% of the time
against rank 0's 20%), so the best resident alternative is frequently an expert
the router rated higher than the one that was lost.

**Implementation**: `ffn_moe_probs` is computed before the argsort that selects
top-k, so the full score vector exists at decision time. `plan()` currently
receives only the chosen ids. Plumbing the score vector through is the change,
after which the stand-in is `argmax(probs masked by residency)` - the
constrained-routing formulation, with no prediction and no new data.

## Anti-co-firing as a probabilistic modifier — tested properly, still fails

Earlier it was tested standalone and deterministic, which was the wrong shape.
Re-tested layered on the existing logic: LRU narrows a 32-candidate sample to
its M stalest, and anti-affinity only breaks that tie. M=1 is pure LRU.

| fill | M | retained | hit | pred tok/s |
|---|---|---|---|---|
| 1 | **1** | **89.62%** | 77.50% | **63.32** |
| 1 | 2 | 89.41% | 77.36% | 63.16 |
| 1 | 4 | 88.82% | 76.70% | 62.11 |
| 1 | 8 | 87.26% | 74.30% | 57.94 |
| 1 | 32 | 73.86% | 59.28% | 33.20 |

Monotonically worse as anti-affinity gets more say. Even the mildest blend
(M=2) is neutral-to-slightly-negative. Recency dominates even when the
alternative signal only has to break a tie.

## Pending work queue (2026-08-25)

**Blocking**
1. The corruption bug above. Until it is understood, no server-measured number
   is trustworthy and the cache cannot be relied on.

**Invalidated, need re-running with the degeneration guard**
2. **[re-measured, 2026-08-29 - the original result does not hold]**
   Admission A/B (-7.6% for removing the gate) - the empirical basis for
   "racing to fill is the wrong move". Re-ran clean with the corruption
   bug fixed: `GGML_CUDA_MOE_CACHE_ADMIT_AFTER=1` (no gate, admit on first
   demand) vs default (`admit_after=2`, gated), same topic-switch 6-prompt
   x4-round methodology.

   | | gated (default) | no_gate (`ADMIT_AFTER=1`) |
   |---|---|---|
   | avg tok/s (n=24) | 55.43 | 54.78 |
   | avg hit rate (n=24) | 79.36% | 79.46% |

   The gap is now ~1%, within noise - the original -7.6% does not
   reproduce. Very likely the original number was itself contaminated by
   the corruption bug's presence at measurement time, not a coincidence
   that the empirical basis for "racing to fill is the wrong move"
   disappears on exactly the number measured before the bug was found.
   The gate is neither clearly harmful nor clearly helpful now - both
   arms statistically indistinguishable here. Consequence for item 4: its
   fill-cost model calibration input is confirmed unreliable, not just
   theoretically provisional.
3. **[closed/superseded, 2026-08-29]** Atlas warming A/B (+0.27 tok/s,
   "neutral"). Not just re-run but thoroughly re-investigated this session
   with a proper multi-round, multi-switch harness - see "Atlas warming,
   second pass" below. Current honest verdict: -0.35pp post-switch pooled
   (roughly neutral, individual requests mixed - one real win, one
   striking regression under investigation), safe (no longer actively
   harmful, unlike two intermediate designs measured at -0.31pp and
   -6.55pp along the way) but not yet a consistent win. The old
   "+0.27 tok/s neutral" number is obsolete, not just unconfirmed -
   superseded by a materially different, more rigorous measurement.
4. Fill-cost model - calibrated from those arms, so every simulation ranking
   that used it is provisional.

**Confirmed, ready to progress**
5. Substitution: +4.7% guarded, quality bounded at +0.47% perplexity. Needs a
   reasoning-benchmark check before shipping; off by default.

**Requested, not yet done**
6. Anti-co-firing as a PROBABILISTIC modifier layered on the existing eviction
   logic (it was only ever tested standalone and deterministic, which is the
   wrong shape).
7. The fold thesis - higher-order 3-way/4-way structure. The test was killed
   mid-run to free RAM and never repeated.
8. Fix the embedding implementation: a trivial common-neighbours baseline
   scores AUC 0.653, the learned embedding 0.490. It should at least match the
   baseline.
9. **[substantially addressed, 2026-08-29]** Integrate a mechanism that
   actually CONSUMES the atlas. Warming touches ~2% of fills, which is why a
   chance-level map and a 0.71 map perform identically on 8099 - no
   integrated atlas comparison means anything until this exists.
   The "Atlas warming, second pass" section below is that mechanism: free-
   slot-only admission ranked by cosine-similarity to a live req_dir
   centroid, burst-widened rank/admit budgets and multi-tensor sweep on a
   detected topic shift, and eviction-side protective weighting
   (moe_cache_weighted_heat) so resident atlas-aligned content resists
   getting evicted without ever being forced. Live /experts stats mid-session
   this pass: prefetches=4178, admission_skips=69201 against
   hits=248992/misses=86600 - roughly 22% of decisions touched the atlas
   path, not 2%. What item 9 actually asked for (an integrated comparison
   that means something) is the multi-round topic-switch benchmark this
   whole section reports, not a separate deliverable - there is no further
   "integration" step left to build; what remains is whether the integrated
   mechanism's real effect is a clear win (still open, see the confidence-
   weighting/lookahead pass below).

**Open research**
10. Coverage-oriented eviction - maximise "some viable stand-in is always
    resident" rather than hit rate. Never tried; the most plausible home for the
    validated 3-way fold structure. See DESIGN ARGUMENT section 4.
11. **[re-opened for a fresh run, 2026-08-29 - the old result is for a
    system that no longer exists]** Perplexity at low residency (1-2%),
    which is where substitution bites. Does NOT need the corruption bug
    fixed. The result most likely to sink the design. "SUBSTITUTION FAILS
    AT LOW RESIDENCY" above (2026-08-25) measured this once already
    (PPL 6.74 off vs 41.32 on, a 6x degradation) - caught on first pass as
    if that closed the item, but it doesn't: that measurement predates
    the rank-gated draft/fallback redesign
    (`GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK`, validated later this
    session at +8.11% tok/s, 4/4 rounds) - substitution unconditionally
    fired on every miss when the 41.32 number was measured. Rank-gating
    exists specifically to bound this kind of quality risk by only
    substituting the router's least-confident picks, never its top ones -
    whether that changes the low-residency verdict has never been tested.
    Re-running with today's substitution mechanism now.
12. Reuse-distance prediction - the 17pp Belady gap. Six signals have failed;
    none of them addresses the actual quantity.
11. **[closed, built 2026-08-29 - see "Co-activation persistence shipped"
    below]** Persist co-activation across runs like `session.history`, for
    cold start. Confirmed via the real code (`moe_cache_predictor_save`,
    `moe-cache.cu:6868-6927`, read in full): no persistence exists for
    `device.co_activation` today - that part holds.
    An earlier version of this note additionally claimed co-activation
    was "the worst arm yet" as a standalone eviction signal and therefore
    not worth persisting - **that claim was wrong, caught on challenge,
    verified by re-reading the actual table** ("six eviction signals"
    section above, 151,580 decisions): plain co-activation ("coact") was
    the SECOND-BEST of five arms tested (56.95 pred tok/s, beating random's
    51.43, losing only to LRU's 63.20) - the "worst arm" description
    belongs to *anti*-co-firing (32.91 tok/s), a different, inverted
    signal, conflated with plain co-activation in the earlier draft of
    this note. Separately, using "underperforms in a session-limited test"
    as an argument against trying persistence has the logic backwards -
    the live-activation predictor (this session's own validated win) was
    itself a *regression* before persistence, and only became a real win
    once cross-restart accumulation gave it more history to work with.
    Whether persisting co-activation would do the same thing here has
    never actually been tested. Genuinely open, not superseded.
12. Track 1.5 - merged, value never independently verified.
13. **[closed, confirmed 2026-08-29]** The cache filling VRAM to a 128 MiB
    floor regardless of `-ngl`; the reserve should scale with the device,
    and the budget should yield under pressure. Already resolved by later
    work, superseding this note: `moe_cache_check_budget`
    (`moe-cache.cu:4301-4397`) computes the reserve as 5% of live free
    VRAM (clamped [128, 1024] MiB, not a fixed 128 MiB - explicitly
    documents the exact scaling problem this item describes: "a 3GB
    reserve left ~47MB available after loading a 26B model" on a 12GB
    card vs "negligible" on a 141GB one), and yields under real pressure
    by releasing dispatch scratch (not pool slabs, which can't be
    partially freed) when `available < committed`. Also fixed, same
    function: `GGML_CUDA_MOE_CACHE_BUDGET_MB` previously being silently
    ignored past the point pools grew beyond it (measured: "BUDGET_MB=256
    and 3413 both peaked at 11,797 MiB - the knob did nothing at all"),
    now reasserted as a hard ceiling on every re-check.

## Open, no proposal yet

From the original table, marked "to be discussed" and not designed further:
embedding + output head weights, compute/activation scratch buffers, CUDA
graph capture state.

## Scoping the three "open, no proposal" VRAM consumers (2026-08-28)

Goal restated precisely, since it shapes which of these are worth building:
the model's footprint stays on NVMe - that's accepted, by design, for any
model whose total size exceeds this hardware's RAM+VRAM. The win condition
is *decode speed* approaching what a dense model sized to the *active*
parameter count would give, not shrinking what has to exist on disk. That
reframing matters below: a component is only a good tiering candidate if it
has the same property moe-cache exploits for experts - real, exploitable
per-token *sparsity* (most of it unused on any given step) plus content that
is *static* across steps (worth caching, not regenerated every time). Not
all three items share both properties equally.

**1. Embedding + output head weights - split into two, only one is a real candidate.**
- *Input embedding table* (`vocab_size x n_embd`, e.g. 154,880 x 4096 for
  glm5next): genuinely the same shape of problem as expert weights. A batch
  only ever touches as many rows as there are tokens in it - a handful, out
  of a 150k+-row table - and a real conversation's actual vocabulary is
  heavily skewed (this codebase already proved the skew is real and
  exploitable: FR-Spec-style draft-vocab trimming, already shipped for the
  MTP drafter's output projection, is the same insight applied one
  component over). **Real, scoped candidate**: an LFRU row-cache for the
  embedding table, mechanically identical to moe-cache's per-expert-row
  cache - key by row index instead of `(tensor, expert)`, evict by the same
  heat/segment logic, admit on miss. Expected win is small in absolute VRAM
  (the whole table is ~1.2 GiB at f32, sub-300 MiB even unquantized-ish at
  low bit-widths) but the *access pattern* match to moe-cache's existing
  machinery is close enough that this is mostly wiring, not new design -
  cheapest of the three to build.
- *Output head / LM head* (same shape, `n_embd x vocab_size`, used the
  other direction): **not the same shape of problem**. Standard sampling
  needs a logit for every vocabulary entry every step - the matmul is
  dense, not sparse, by construction (this is exactly why FR-Spec-style
  trimming for the drafter's own head required accepting a bounded vocab
  subset and measuring the real throughput-vs-VRAM trade rather than
  getting the win for free - see `docs/index.html`, "FR-Spec draft-vocab
  trim... -26%"). Tiering this the moe-cache way would mean recomputing
  which vocabulary rows are "hot" doesn't help, since ALL rows are read
  every step regardless of recent history. The only way to make this
  sparse is to change what gets computed (a restricted/approximate output
  projection), which is a sampling-behavior decision, not a caching one -
  out of scope for "the same tiered regime", genuinely a different project.
  Given its size is comparable to the embedding table, defaulting to fully
  VRAM-resident (no tiering) is the correct baseline unless VRAM pressure
  specifically forces the question.

**2. Compute/activation scratch buffers - does not fit the cache model at all.**
moe-cache's LFRU works because expert weights are the same bytes on every
hit - a slot holds one expert's data indefinitely until evicted, and a hit
means "the bytes I need are already here." Scratch/activation buffers
(`d_act`, `d_act_q8`, `d_out` in moe-cache.cu; the scheduler's own reserved
compute buffer, e.g. "441.00 MiB" logged this session) are overwritten
every layer, every token - there is no stable content to hit or miss on,
so "cache hit rate" isn't a meaningful concept here at all. The real lever
for this component is a completely different kind of optimization:
*right-sizing and reuse*, not tiering - e.g. whether sequential layers can
share one scratch region (only one layer computes at a time) instead of
each layer reserving its own, or whether the scheduler's worst-case
reservation is looser than the real steady-state need. Smaller expected
win than the other two (this is working memory, not multi-gigabyte
weight data), and it needs its own investigation into the scheduler's
buffer-reuse logic - **recommend deprioritizing this one**; it was grouped
with the other two by association ("also consumes VRAM"), not because it
shares moe-cache's actual mechanism.

**3. CUDA graph capture state - a real, different-shaped candidate; already half-cached by construction.**
A captured CUDA graph *is* already a cache entry, structurally - captured
once per distinct graph topology/shape, replayed cheaply thereafter
("graphs reused = 98" in this session's server logs is literally a hit
counter). The open question isn't "should this be cached" (it already is,
implicitly, for the lifetime of the process) but **"should it be bounded
and evictable"**: if a server sees many distinct batch shapes over its
lifetime (varying ubatch sizes, concurrency levels, speculative-decode
draft widths), each distinct shape accumulates its own captured graph
object, and nothing currently bounds how many accumulate or reclaims a
graph for a shape that stopped recurring. This *does* fit the moe-cache
shape reasonably well: key by graph topology signature instead of
`(tensor, expert)`, heat/recency-track hits the same way, evict the
coldest captured graph under memory pressure rather than never evicting.
**Real, scoped candidate**, but lower urgency than the embedding cache
unless a concrete case of unbounded graph-object growth is actually
observed on real traffic - worth a measurement pass (how many distinct
shapes does a real workload actually produce, and how much VRAM does that
accumulate to) before assuming eviction is needed at all.

**Net scoping verdict**: of the three, the embedding-row LFRU cache is the
strongest, cheapest, most directly analogous win - build that one first if
this track gets picked up. CUDA graph capture eviction is real but should
be measured (not assumed necessary) before building. Compute/activation
scratch buffers should be re-scoped as a buffer-reuse investigation, not
carried forward as a "tiered cache" item - it doesn't fit the model.
Output head weights are explicitly descoped from tiering; any future win
there is a sampling-side project, not a caching one.

**CORRECTION (2026-08-29, track-2): item 3 was already closed upstream,
before this scoping note was written.** `ggml_backend_cuda_context::cuda_graph()`
(`ggml/src/ggml-cuda/common.cuh:1428-1453`) already keys captured graphs
by `first_node_ptr` (one graph object per distinct topology/shape, not a
single global graph - from upstream `81ab64f3c` "enable cuda-graphs for
n-cpu-moe") and already evicts: a sweep every 5s drops any graph unused
for >=10s (from upstream `b94050e89` "use LRU based eviction for cuda
graphs"). Both merged into this tree already. No measurement pass, no new
mechanism needed - this item is done, not "lower urgency." Track-2 scope
narrows to the other two: embedding-row GPU cache, and scratch-buffer
right-sizing.

## Already closed out (adjacent work, same session)

- Readahead re-test: re-measured properly with an interleaved harness under
  today's full stack. Real result — now consistently *worse* on cold-start
  (~+21%, not the originally documented improvement), roughly neutral on
  warm throughput. Not shipped as a default change; documented honestly.
- `GGML_SCHED_PREFETCH_EXPERTS` VRAM wiring: real measured cost (820 MiB)
  reserved additively in `fit.cpp`'s margin. Commit `3417a616b`. Feature
  itself still ships off by default — this only makes it safe to opt into.

## Fast, reliable, content-independent reproducer found (properly verified)

Three days of chasing a single-prompt, single-sequence trigger just gave way
to something much simpler: **4 concurrent long-generation requests, on
completely unrelated benign prompts, corrupt reliably at full speed with no
special prompt content at all.**

Verified twice, the second time with full rigor (fresh restart, PID change
confirmed, a single request checked clean BEFORE the real test, so this is
not carrying state from a prior arm):

```
--moe-cache 512, fresh server, born-clean confirmed, then 4 concurrent
requests (Roman Empire essay / how computers work / water cycle / mountain
journey, 500 tokens each, --parallel default 4):

A slot=0 elapsed=19.31s -> ////////... (DEGEN)
B slot=1 elapsed=19.31s -> ////////... (DEGEN)
C slot=2 elapsed=19.31s -> ////////... (DEGEN)
D slot=3 elapsed=19.31s -> ////////... (DEGEN)
```

All four slots, all four completely different prompts, corrupted at the
**same elapsed time to the hundredth of a second**. That is not four
independent failures - it is one shared piece of state breaking and
instantly poisoning every concurrently-batched sequence at once.

**A single 4-way SHORT concurrent burst (64 tokens each) does NOT reproduce
it** - same prompts pattern, same slot spread, all four came back clean. So
short concurrency alone is insufficient; sustained/longer concurrent
exposure is required. This points at a genuine race whose window is normally
too narrow to hit, opened wide enough by either (a) heavy instrumentation
slowdown (compute-sanitizer reproduced it on a single BENIGN short prompt,
no concurrency needed at all - see below) or (b) enough real concurrent
decode duration at full speed.

**Directly ties to the earlier TSan finding**: `ggml_compute_forward_mul_mat_id`
(ggml-cpu.c) runs a two-phase quantize-then-dot-product pattern against a
SHARED `wdata` scratch buffer covering the WHOLE combined ubatch when
multiple sequences are batched together (continuous batching's normal
behavior). TSan flagged a real race between the quantize-write phase
(`quantize_row_q8_K`) and the dot-product-read phase
(`ggml_vec_dot_q4_K_q8_K`) inside this exact function, though a structural
read of the barrier between them (ggml-cpu.c:~1806) didn't reveal how it
happens, and a same-thread reentrancy guard around the two moe-cache lock
sites the TSan stack named found zero violations - so the mechanism is not
confirmed, but the SHARED-BUFFER-ACROSS-CONCURRENT-SEQUENCES shape is exactly
consistent with "one race hit corrupts every sequence in that ubatch
simultaneously," which is precisely what this reproducer shows.

**Also reproduced under compute-sanitizer with NO concurrency needed at
all**: a single benign "Name three primary colours" request, run alone
(sequential, not concurrent) under `--tool memcheck`'s heavy per-kernel
instrumentation, produced literal `////` output too (verified against raw
response content, not just the classifier). Memcheck itself found zero
out-of-bounds/uninitialized-read violations through that run - so whatever
this is, it isn't the specific violation classes memcheck checks for. This
independently confirms that TIMING ALONE (no concurrency) is also
sufficient, given enough distortion - concurrency and slowdown are two
different ways of reaching the same underlying race.

**Next steps**: (1) use this fast reproducer (~20s, not minutes) to iterate
far faster than the old sequential trigger ever allowed - re-run every
existing instrument (NAN_PROBE, IDENTITY routing probe, GROWTH_LOG,
COLLECT_BOUNDS) against 4-way concurrent long generation instead of the slow
single-sequence trigger; (2) chase the ggml-cpu.c mul_mat_id barrier logic
specifically under REAL multi-sequence concurrent load (not warmup, not
sequential) - the TSan report analyzed so far was largely from warmup and a
single sequential trigger run, never from genuine concurrent decode traffic
hitting the shared wdata buffer this reproducer implicates directly.

## Kernel-level finding: one confirmed NVRM page-table allocation failure

`dmesg` shows a genuine NVIDIA driver-level fault during this session's heavy
testing: `[24863.131786] NVRM: failed to allocate page table!` - corresponding
to wall-clock 2026-08-25 18:41:39, during the earlier VRAM-exhaustion /
TSan / compute-sanitizer testing window (not correlated with the later
concurrent-reproducer runs, which happened after 23:00 the same day).

This is a real, driver-level allocation failure, not an application bug -
worth keeping in mind since a failed page-table allocation for ANY GPU
context sharing this device (ours or otherwise) could plausibly leave that
context in a state where later kernels read garbage without any bug in our
own code. Not established as the cause of the reproducible corruption above
(no timestamp correlation with the concurrent reproducer's actual failures),
but a genuine anomaly this investigation hasn't otherwise explained.

Separately: a felt ~1s system-wide input freeze (mouse/keyboard) during the
4-way concurrent reproducer left no corresponding kernel-log entry. Most
likely explanation is mundane GPU command-queue contention between our
compute workload and the desktop compositor sharing one physical GPU under
heavy concurrent load - a real side effect of the load, not necessarily
evidence of the same mechanism as the corruption itself.

## New session (2026-08-26): the scratch-buffer guard instrumentation had its own bug

Picked up with the uncommitted working tree already carrying the next round of
instrumentation from the prior session: per-slot guard bytes in the weight
slab, poison-on-growth for the dispatch scratch buffers (`d_act`, `d_act_q8`,
`d_out`), a slot `generation` counter for `GGML_CUDA_MOE_CACHE_GEN_CHECK`, and
an always-on circuit breaker that rejects non-finite `collect()` output and
forces the existing CPU fallback.

**First run of the 4-way concurrent reproducer (cache 512, `-ncmoe 30`,
`--parallel 4`, 500 tokens x4) fired 20 "SCRATCH GUARD BREACH" warnings** -
looked at first like the smoking gun the prior session was chasing. It
wasn't: the guard check assumed the bytes right after a dispatch call's
`used` size are always still-canary, but the canary is only laid down once,
at buffer (re)growth, not before every dispatch. These scratch buffers are
reused across calls with varying `n_hits`, so a smaller call following a
larger one leaves the PRIOR call's real, legitimately-written data sitting
past its own `used` boundary - not canary. The check was flagging that
leftover data as a false "breach." All 20 breaches were this false positive
(0 real per-slot `GUARD BREACH`, 0 `OUTPUT REJECTED` non-finite hits in that
run) - output came back clean 4/4 only because the false breach still forces
`ok=false`, which still correctly triggers the (already-safe) CPU fallback.

**Fixed**: added a high-water-mark per scratch buffer (`device.d_act_hwm`,
`act_q8_hwm`, `d_out_hwm`), reset to 0 on regrowth, and gated the guard check
on `used >= hwm` - i.e. only checkable when a call reaches a fresh deepest
point since the last growth, the one case where the region past `used` is
provably still pristine canary. `ggml/src/ggml-cuda/moe-cache.cu` around the
`device.d_act` etc. struct fields and in `moe_cache_dispatch`/`moe_cache_collect`.

**Re-ran with the fix**: two full 4-way concurrent bursts (8 completions,
500 tokens each) - 0 scratch breaches, 0 real slot breaches, 0 non-finite
rejections, all 8 outputs clean. Also ran 4 sequential single-prompt
64-token generations AND one full 4-way concurrent 300-token burst under
`compute-sanitizer --tool memcheck` (the tool that reproduced corruption with
zero concurrency in the prior session) - sequential runs came back clean with
zero memcheck violations; the concurrent-under-memcheck run was still in
progress as of this note (memcheck's per-kernel instrumentation makes it very
slow - each run takes minutes instead of seconds). This is NOT yet strong
enough evidence to call the underlying corruption fixed - the original bug
was already known to be intermittent and this session has not yet reproduced
it even once post-fix, sequential or concurrent, sanitized or not, which is
consistent with either "actually fixed" or "still there but not hit yet."

**Also gated the new guard-check instrumentation behind
`GGML_CUDA_MOE_CACHE_GUARD_CHECK`** (on by default for now): the per-slot and
per-scratch-buffer guard sampling costs `n_hits + 3` extra `cudaMemcpyAsync`
launches per `collect()` call - real per-call driver overhead on every MoE
layer of every token, even though they share the existing stream sync. Worth
paying while this bug is still being actively chased; not something a
production hot path should carry once it's closed out. Set to `0` to disable
both guard checks while keeping the (cheap, always-on) non-finite output
rejection.

**Code-review pass (requested, not bug-triggered)**: read through the slot
state machine, pin/generation lifecycle, and `moe_cache_grow_device`'s
free/realloc path looking for a structural bug independent of the reproducer.
One real latent issue found, not (yet) implicated in the corruption:
`moe_cache_slot_reset` (`moe-cache.cu:1557`) unconditionally zeroes
`slot.readers` even when it's positive - the function's own comment already
documents this as "the pin discipline breaking... invisible afterwards
because the refcount is gone." `GGML_CUDA_MOE_CACHE_PIN_AUDIT=1` reported zero
violations through a corrupting generation in the prior session, so normal
eviction (`moe_cache_pick_coldest_unpinned`) never actually selects a
pinned slot in practice - the bug is latent, not (so far) reachable. Left
as a documented risk rather than changed, since hardening it (refuse to
reset a slot with `readers > 0`, or assert) means deciding what every caller
should do instead when eviction has no unpinned candidate, and that's a
design call, not a one-line fix. Worth doing before shipping, not blocking
right now.

`moe_cache_grow_device`'s unconditional `cudaFree(*pointer)` on the old
buffer (the `GGML_CUDA_MOE_CACHE_GROWTH_SYNC` flag that adds an explicit
`cudaStreamSynchronize` first is opt-in, default off) looked suspicious on
first read, but traced out as safe: `moe_cache_grow_device` is only called
from `moe_cache_dispatch`, itself only reachable while `device.dispatch_mu` is
held for a node's whole `begin()..end()` lifetime, and `collect()` always
`cudaStreamSynchronize`s before that lock is released - so by the time the
*next* dispatch's growth call runs, everything that could have been reading
the old buffer is already provably finished. Also, plain `cudaFree` (as
opposed to the stream-ordered `cudaFreeAsync`) is itself a synchronizing
call in CUDA's runtime semantics. Not a bug; ruled out by tracing rather than
by testing this time.

### Broader review, fanned out (2026-08-26): two real bugs found and fixed elsewhere in the codebase

User asked for a review of the whole codebase for bugs, not just ones tied to
the corruption chase. Given the scale, fanned this out to four parallel
review agents (ggml-cuda kernels, ggml-cpu core, `src/` llama core, server +
common) rather than a serial read. Two came back with genuine, independently
verified bugs; the other two (ggml-cpu core, `src/` llama core) found nothing
they could back with a concrete trigger after real effort (see their reports
if this needs re-litigating - both explicitly checked the highest-risk areas:
threadpool/wsize sizing in ggml-cpu.c, VMM commit + host-pinning register/
unregister rounding in llama-mmap.cpp/llama-kv-cache.cpp).

**Fixed: `ggml-cuda/mmid.cu`, `mm_ids_helper`, lines 47 and 72 - a
previously-fixed bug had regressed back in.** `nex_prev += expert_used <
expert;` counts every expert index below the current one to find where this
expert's compact-array slice starts - but `-1` (the "this slot is skipped"
sentinel id, general ggml convention, ggml-cpu.c's own `mul_mat_id` already
handles it by zeroing the dst row) satisfies `-1 < expert` for every
non-negative `expert`, so a single skipped id inflates `nex_prev` for every
expert block, opening a phantom unwritten region at the head of the compact
`ids_src1`/`ids_dst` arrays that downstream MMQ/MMF quantize kernels then
read as garbage row indices - an illegal memory access. This exact bug was
already fixed once, in commit `e4ed3cdcf` (`expert_used >= 0 &&`), but a
later upstream sync (`5839ba352`, "CUDA: dedup MoE gate/up activation
quantization") silently reintroduced the file without that fix. Reapplied
the same guard at both sites.

Also confirmed while investigating this: that same sync dropped the ENTIRE
feature the fix belonged to, not just the one guard - `d5b1a815c` ("id=-1
skip support in CUDA mul_mat_id fast paths") added a
`mm_ids_zero_skipped_rows` kernel (mmid.cu/mmid.cuh) plus call sites in
mmq.cu, mmf.cu and mmvq.cu (including an early-return in `mul_mat_vec_q`/
`mul_mat_vec_q_moe` on `id < 0`, since without it a skipped id is used
directly as a channel index - an out-of-bounds read). None of that is in the
tree any more (`grep mm_ids_zero_skipped_rows` across ggml-cuda/*.cu is
empty). Left AS IS rather than reconstructing the whole feature: grepped the
whole codebase for what would have to produce a `-1` id on the CUDA dispatch
path (the "hot/cold expert pack split" the commit messages describe) and
found nothing - no `expert_pack`/`hot_cold` identifiers anywhere outside one
comment in moe-cache.cu, and `llama-graph.cpp`'s actual MoE routing
(`ggml_argsort_top_k` on real logits) never emits `-1`. So the regression is
real but currently unreachable; the `mm_ids_helper` bounds fix alone is
correct, minimal, and safe to ship on its own. If the hot/cold pack feature
is meant to come back, reconstructing `mm_ids_zero_skipped_rows` and its
call sites is a separate, deliberate follow-up.

**Fixed: `tools/server/server-context.cpp:5506`, rerank endpoint, unvalidated
client `top_n`.** `int top_n = json_value(body, "top_n", ...)` took the
client's JSON value with no lower-bound check, then
`format_response_rerank` (`server-common.cpp:1420`) does
`elements.resize(std::min(top_n, (int)elements.size()))` - a negative
`top_n` (e.g. `"top_n": -1`) makes `std::min` pick the negative value, which
`resize()` then implicitly converts to a huge `size_t`, throwing
`std::length_error` or attempting a multi-exabyte allocation. Remotely
triggerable: any `POST /rerank` or `/v1/rerank` request with a populated
`documents`/`texts` array and a negative `top_n`. Fixed by clamping at the
parse site: `std::max(0, json_value(...))`.

Both fixes rebuilt clean (`ggml-cuda`, `server-context` targets).

**Closing this session's corruption-reproduction attempts.** The 4-way
concurrent 300-tokens-each burst under `compute-sanitizer --tool memcheck`
(the same tool that reproduced corruption with zero concurrency in the prior
session) finished: all 4 outputs clean (uniq_ratio 0.038-0.052, well above
the ~0.02 corrupted threshold), 0 memcheck violations, 0 circuit-breaker
triggers of any kind. Tally for this session, all post-guard-fix: 2x native
4-way concurrent bursts (8 completions), 4x sequential single-prompt runs
under memcheck, 1x 4-way concurrent run under memcheck (4 completions) - 16
completions total, 0 corrupted, 0 sanitizer violations. This is evidence the
fixed instrumentation + the two unrelated bugs fixed elsewhere did not
obviously make things worse, and is consistent with (not proof of) the
underlying race being fixed - the original bug was already known to be
intermittent, and this session found and fixed a real bug in the
*instrumentation* meant to catch it, not a confirmed fix to the corruption's
own root cause. The mechanism remains formally unconfirmed. Recommend: watch
for recurrence in normal use with `GGML_CUDA_MOE_CACHE_GEN_CHECK=1` and the
guard checks left on; if it recurs, the fixed guards should now report
real breaches instead of false ones.

**Follow-up review pass, same "silently-reverted fix" pattern**: a second
agent checked `fattn.cu`, `rope.cu`, `cpy.cu`, `softmax.cu`, `norm.cu`,
`binbcast.cu`, `concat.cu`, `common.cuh` against every fix commit touching
them in this fork's history - all six most plausible candidates (grid-size
overflow guards, KQ-mask offset int64 casts, the cpy transpose double-buffer
race fix, rope's non-contiguous stride fix, the block_reduce SMEM-reuse
race fix) are still intact in the current tree. No second instance of the
`mmid.cu`-style regression found in this scope.

**Git-archaeology pass found three more instances of the exact same
regression**, all lost in the same `5839ba352` ("CUDA: dedup MoE gate/up
activation quantization") upstream-sync overwrite that dropped the mmid.cu
fix. All three verified directly (read the original fix diff, confirmed the
pre-fix pattern is what the current file has) and restored, rebuilt clean:

- **`ggml-cuda/mmq.cu`** (fix `2fc72679d` + the earlier `e4ed3cdcf` sentinel-fill
  it built on): the two `cudaMemsetAsync` calls that sentinel-fill
  `ids_src1` (0xFF) and zero `ids_dst` before `mm_ids_helper` runs were both
  gone - any -1 id left the compact arrays' unwritten tail holding pool
  garbage, which the quantize/mm kernels would then read as a row index.
  Restored both.
- **`ggml-cuda/mmvf.cu`** (fix `a3f26d8b5`): the kernel-side
  `if (ids && channel_x < 0) return;` guard in `mul_mat_vec_f` was gone - a
  -1 id would be used directly as a channel index (OOB read) instead of
  skipping the row. Restored the guard. Did NOT restore the fix's other half
  (the host-side `ggml_cuda_launch_mm_ids_zero_skipped_rows` call that
  pre-zeroes skipped rows for correctness) since that kernel itself is part
  of the still-missing `d5b1a815c` feature (see above) - restoring the
  memory-safety guard alone is enough to remove the OOB read; the dst row
  for a skipped slot is just stale instead of zeroed, same tradeoff already
  accepted for mmid.cu/mmvq.cu.
- **`ggml-backend.cpp:1926`** (fix `8a14a58b3`,
  `ggml_backend_sched_compute_splits`'s used-experts scan): had regressed
  back to `GGML_ASSERT(id >= 0 && id < n_expert)` with no negative-id skip.
  Unlike the CUDA kernel-side findings, this one is NOT gated behind the
  fork's dead hot/cold-pack feature - it scans the same `ids` tensor that
  `ggml-cpu.c`'s own `mul_mat_id` already treats -1 as a normal, expected
  sentinel (upstream ggml convention, not fork-specific), so any path that
  legitimately produces a -1 id and also goes through backend scheduling
  would abort the whole process on this assert. Restored the `if (id < 0)
  continue;` skip before the assert.

All three rebuilt clean (`ggml-cuda`, `ggml` targets), then a full
`llama-server` relink + smoke test (model load, one completion, one rerank
request against a non-reranking server to confirm it still error-responds
rather than crashing) passed. Total for this session: 5 real, independently
verified bugs found and fixed (1 instrumentation false-positive, 1
`mmid.cu` bounds regression, 1 rerank DoS, 2 more instances of the same
mmid.cu-class regression in mmq.cu/mmvf.cu/ggml-backend.cpp), all outside
the corruption bug's own root cause, which remains unconfirmed.

## CORRUPTION — live reproduction, caught and localized in real time (2026-08-26, later)

While running the substitution reasoning-benchmark (item 5, see below), the
`--moe-cache 512 -ncmoe 30` server it was hitting degenerated to `////...`
on essentially the FIRST request served after a fresh restart, deterministic
on the same prompt every time re-tried. This is a far faster, simpler
reproducer than anything found previously - a single non-concurrent
`/completion` request, no compute-sanitizer needed, no accumulated churn.
Investigated live against this exact reproducer with the tooling this
session already built plus new instrumentation, all findings direct not
inferred:

**The circuit breaker fires but does not save the output.** Every rejected
row gets recomputed on CPU (the existing fallback) and is *still* wrong -
proof this is real data corruption, not a one-off kernel glitch the retry
already covers.

**Weight bytes are NOT mutated.** `GGML_CUDA_MOE_CACHE_WEIGHT_GUARD=1`
re-hashed 3082 admitted experts' SOURCE bytes against their admission-time
hash on this exact reproducer: 0 mutated, every sweep. Rules out "something
overwrites the source weight file's mapped pages."

**No fill-vs-pinned-read race.** `GGML_CUDA_MOE_CACHE_GEN_CHECK=1` (compares
each pinned slot's generation counter before/after the mandatory
post-dispatch sync) and `GGML_CUDA_MOE_CACHE_VERIFY_DISPATCH=1` (checks every
slot is still `valid`/pinned at dispatch time): 0 violations, both, on this
exact reproducer. Rules out the fill worker rewriting a slot out from under
an in-flight read - the leading theory since early in this investigation.

**The activation feeding the corrupted layer is already NaN going in - not
the dot-product output, the INPUT.** `GGML_CUDA_MOE_CACHE_NAN_PROBE=1` (an
existing check in `ggml-cpu.c` that had apparently never actually been
enabled in this investigation - it was silently gated behind its own env var
this whole time) shows `src1` for `blk.22.ffn_gate_exps`/`ffn_up_exps` is
**100% NaN (8192/8192 or 2048/2048 elements, every run)**, first appearing at
exactly layer 22, every earlier layer (0-21) clean. A first attempt at
measuring this got a false "amax=0" reading - `std::max` silently ignores
NaN comparisons (`NaN < x` is always false), so the original diagnostic was
blind to the very thing it was trying to catch. Fixed by counting NaN
elements explicitly rather than trusting `max`'s behavior around them
(`ggml-cuda/moe-cache.cu`, the new `GGML_CUDA_MOE_CACHE_ACT_MAGNITUDE_LOG`
debug knob). This rules out an FP16 quantizer-scale-overflow theory that
looked promising for the same reason (checked: real activation magnitudes at
clean layers 20-21 are single-digit, nowhere near overflow range) - the
input isn't merely small or zero, it is NaN, before moe-cache's own
quantizer ever touches it.

**`--moe-cache off` is clean on the identical prompt** (confirms: this is
still a real moe-cache-specific bug, not a pre-existing model/quantization
defect that any -ncmoe config would hit) - the model correctly computes
24-5=19 with the cache off. So something about the cache being ENABLED
causes an EARLIER computation (attention or norm at layer 22, before its own
FFN/MoE block - `ffn_gate_exps`'s `src1` is layer 22's POST-ATTENTION
normed hidden state, not layer 21's FFN output directly) to already be NaN.
No `OUTPUT REJECTED` fires for any layer below 22, so whatever moe-cache
itself dispatches at layers 0-21 is fine - the corruption's true origin is
somewhere in layer 22's own attention block or norm, upstream of any
moe-cache dispatch for that layer, but only when moe-cache is active
elsewhere in the process.

**UPDATE, same session: by far the best reproducer found in either
investigation session.** Chasing the "requires `llama-server`" finding
further with bisection, in order of what was ruled out:

- `GGML_CUDA_DISABLE_GRAPHS=1`: still corrupts. Not CUDA graphs (consistent
  with the earlier graphs-on/off A/B in this same doc, now confirmed a third
  way).
- `--no-warmup`: the FIRST real request after startup is now clean (fixed
  the "corrupts on request 1" framing above) - but this was a red herring
  about warmup specifically, not a fix. See next point.
- **The real trigger: exactly TWO sequential, non-concurrent requests, where
  the second uses a DIFFERENT prompt than the first.** With `--no-warmup
  --parallel 1` (single slot, zero multi-slot/unified-KV-cache complexity,
  zero concurrency): request 1 (pencils word problem) clean, request 2 (a
  *different* word problem, garden area) corrupts, every time, deterministic.
  But sending the SAME prompt three times in a row on a fresh server stays
  clean all three times (uniq_ratio identical across all three). So the
  trigger isn't "any second generation" - it's specifically a *change* in
  what's being generated, i.e. new/different expert routing demanding
  admissions the first prompt's residency doesn't already cover. This is
  the same "requires admission churn" shape the very first investigation
  session already suspected, now pinned to an exact, tiny, two-request
  recipe instead of needing sustained concurrent traffic or
  compute-sanitizer's timing distortion to open the window. Warmup's
  earlier-observed effect was just this same mechanism: warmup is itself an
  initial (different-content) decode, making the first REAL request "request
  2" relative to it - consistent, not a separate cause.
- Multi-slot/unified-KV-cache machinery is NOT required: still corrupts
  with `--parallel 1` (n_slots=1). Rules out slot-reuse/LCP-similarity
  logic as the mechanism, despite superficially fitting ("different prompt
  -> different KV reuse pattern").
- Scheduler/graph rebuild is NOT involved: `sched_reserve()`'s "reserving
  ..." log (which fires on every `ggml_backend_sched` teardown/rebuild -
  see `llama-context.cpp`, `sched_need_reserve`) never appears between the
  two requests. Rules out the moe-cache session take/adopt-across-rebuild
  path (`ggml_backend_sched_take/adopt_moe_cache_session`) as the mechanism,
  despite superficially fitting ("state carried across a rebuild boundary").
- Full diagnostic stack (`GEN_CHECK`, `VERIFY_DISPATCH`, `WEIGHT_GUARD`,
  `PIN_AUDIT`, `VERIFY_SLOTS`, `VERIFY_ROWS`, `MAP_AUDIT`) run together on
  this exact 2-request reproducer: all silent except `NAN_PROBE` and the
  existing circuit breaker. Every moe-cache-internal consistency check this
  investigation has ever built says the cache's own bookkeeping is correct
  at the moment of failure - again.
- On this run `NAN_PROBE` caught it even earlier than before: `blk.1`
  (the SECOND transformer layer), not `blk.22` - the exact failing layer
  varies by run/prompt, but it's always the first MoE layer whose `src1`
  gets checked after whatever actually broke, i.e. the true origin is
  upstream of moe-cache's own dispatch call for that layer (that layer's own
  attention/norm, or earlier), not in moe-cache's kernel math - reinforced,
  not just repeated, by seeing it move.

**Net position**: two full investigation sessions have now independently
and exhaustively ruled out every piece of moe-cache's own bookkeeping (pin
lifecycle, generation counters, slot state machine, weight-byte integrity,
map injectivity, row accounting, dispatch-time slot validity) as the
mechanism, while conclusively establishing (a) it requires moe-cache
enabled, (b) it requires a real request whose expert routing differs from
what's already resident (i.e. real admission/eviction churn, confirmed
today down to a 2-request minimum), and (c) the actual corrupted value is
the ACTIVATION flowing into an early MoE layer, not any weight or cache
data. The most promising unexplored surface given this: whatever moe-cache
code path runs *during eviction/admission itself* while a subsequent
ubatch's attention/norm computation for an EARLY layer is proceeding
concurrently on the main thread - i.e. the background fill worker doing
real work (not just idling) at the same time the main thread is past
layer 0/1's attention, before either side's own consistency checks would
catch anything, since none of today's checks instrument attention or KV
cache at all. Next session: extend `NAN_PROBE`-style checking (or the
`common_debug_cb_user_data abort_on_nan` mechanism now built into
`llama-eval-callback`, extended to also run TWO different prompts back to
back rather than one, to get the same clean-CLI-tool control this session
already established for the single-prompt case) into attention/KV-cache
tensors specifically, to catch the corruption at its actual origin instead
of one or more layers downstream of it.

## BREAKTHROUGH, same session: reproduced in a single CLI process, exact op identified

Did the "next step" above immediately rather than deferring it. Extended
`examples/eval-callback` further: `run()` now takes a prompt string
parameter, and after the first prompt's greedy generation loop, if
`EVAL_CALLBACK_PROMPT2` is set, resets the KV cache for sequence 0
(`llama_memory_seq_rm(mem, 0, 0, -1)`) and runs a second, different prompt
through the identical `abort_on_nan` instrumentation, in the same process.

**First attempt used `llama_memory_seq_rm(mem, -1, -1, -1)`** (wildcard
"clear everything") and hit a DIFFERENT, unrelated crash:
`GGML_ASSERT(cell.has_seq_id(seq_id)) failed` in
`llama_memory_recurrent::find_slot` (`llama-memory-recurrent.cpp:575`). This
model (`Ornith-1.5-35B`) turns out to be a **hybrid SSM/attention/MoE
architecture** (Gated DeltaNet linear-attention layers with their own
`llama_memory_recurrent` conv/SSM state cache, wrapped in
`llama_memory_hybrid` alongside the normal attention KV cache -
`cache_r_l0`/`cache_s_l0` per-layer state tensors, `GATED_DELTA_NET`/
`SSM_CONV` ops) - not a plain transformer. The wildcard seq_rm doesn't
correctly reset this recurrent component for this architecture; a real bug
in its own right, separate from the one being chased, not investigated
further this session (noted as pending-work-worthy below). Using the
specific sequence id instead (`seq_rm(mem, 0, 0, -1)`, seq 0 for slot
0 - what `llama-server` actually calls to release a slot between requests)
avoided it cleanly.

**With the correct reset, the bare CLI tool reproduces the corruption
exactly**: prompt 1 (pencils) generates 40 clean greedy tokens, KV reset,
prompt 2 (garden area) - and the process aborts with "encountered NaN"
almost immediately, during PREFILL of the second prompt, not generation.
Full single-process repro, zero server, zero HTTP, zero slot machinery,
zero concurrency, zero compute-sanitizer.

**The exact failing tensor, read directly off the callback trace**:
`ffn_moe_down-0` - `blk.0.ffn_down_exps`'s `MUL_MAT_ID`, THE VERY FIRST MoE
op of the very first transformer layer of the second prompt. Everything
printed before it in that prompt's graph is clean (no premature abort,
meaning every earlier tensor's sum was finite): the embedding lookup, the
FULL Gated-DeltaNet linear-attention block (conv state read/update,
`SSM_CONV`, `GATED_DELTA_NET`, ssm state read/update, gating, output
projection, residual add), the post-attention norm, the router logits/
softmax/argsort/top-k, `ffn_moe_gate-0`, `ffn_moe_up-0`, and
`ffn_moe_swiglu-0` (the SwiGLU-combined gate*up product - i.e. `down`'s own
input). **`ffn_moe_down-0`'s input (`ffn_moe_swiglu-0`) is verified clean
and its weight (`blk.0.ffn_down_exps.weight`) is bytes this whole
investigation has repeatedly verified never mutates** - yet its output is
NaN. This is the most direct, unambiguous evidence yet that the defect is
IN moe-cache's own dispatch/kernel path for this call, not something
upstream feeding it bad data, and not attention/KV-cache/SSM-state related
despite how promising that surface looked from the server-side symptoms.

**Confirmed immediately (checked via `gguf-py`, `GGUFReader`) - this is the
session's most actionable result: `blk.0.ffn_down_exps.weight` is
`Q6_K`. `blk.0.ffn_gate_exps.weight` and `blk.0.ffn_up_exps.weight` (which
dispatched cleanly, same call, same layer, same cache) are both `Q4_K`.**
Standard for a `_M` quant mix (down-projections commonly get extra
precision), and it means the failure isolates to the Q6_K-specific dispatch
path - `vec_dot_q6_K_q8_1`, or something in how `moe_cache_allocate_pool`
computes `slot_stride` for Q6_K's block layout (210 bytes/block, QK_K=256,
vs Q4_K's 144 bytes/block - the `slot_stride` guard-rounding comment in
`moe-cache.cu` explicitly calls out Q4_K's block size as the reasoning
example; worth checking it was verified against Q6_K too, not just derived
generically and assumed correct) - not the shared MMVQ machinery in
general, and not moe-cache's dispatch/collect bookkeeping (identical for
every wtype). This is a small, closed-off surface: one kernel
(`vec_dot_q6_K_q8_1` / `vec_dot_q6_K_q8_1_impl_mmvq`, `vecdotq.cuh`) and one
allocation-time calculation (`moe_cache_allocate_pool`'s `slot_stride`
derivation, `moe-cache.cu`), both easy to re-check specifically for Q6_K's
constants next session, with a reproducer that now takes seconds, not
minutes, to re-run.

**Important refinement, checked immediately after the type finding**: `down`
isn't just a different quant type from `gate`/`up` - it's a **different,
transposed SHAPE**: `ffn_down_exps.weight{512, 2048, 256, 1}` (n_in=512,
n_out=2048) versus `ffn_gate_exps`/`ffn_up_exps.weight{2048, 512, 256, 1}`
(n_in=2048, n_out=512) - down projects the 512-dim SwiGLU intermediate back
to the 2048-dim residual stream, the reverse of what gate/up do. Different
shape means a different `expert_size`, which means `moe_cache_find_pool`
(keyed on `(expert_size, wtype)`) puts `down` in a **completely separate
pool** from `gate`/`up`, with its own independent slot array, admission
history, and `slot_stride`. `QI6_K`/`QR6_K` (both 32/2, from
`ggml-common.h`) are structurally identical to `QI4_K`/`QR4_K`, so the
q8_1-activation-block bookkeeping the kernel walks is not obviously
type-sensitive - which weakens "Q6_K's kernel math itself" as the sole
explanation and strengthens "whatever's specific to `down`'s pool" (its
shape, or simply being a separate, independently-admitted pool from
gate/up's) as at least as likely a culprit. Both quant type AND shape/pool
identity differ for `down` versus the two tensors that dispatched cleanly
in the same call - this session did not have time to isolate which (or
both) actually matters; that is the concrete next step, e.g. testing
against a hypothetical quant where down and gate/up share both type AND
shape, or adding pool-identity/shape logging to the existing NAN_PROBE /
OUTPUT REJECTED instrumentation.

**Remaining testable next steps, none done yet this session (time
constrained)**:
1. Check `vec_dot_q6_K_q8_1_impl_mmvq` (`vecdotq.cuh`) line by line for
   anything Q6_K-specific that could produce non-finite output from finite
   inputs - a division, an unguarded scale multiply, an index computed
   differently than Q4_K's equivalent. Cross-check against
   `moe_cache_allocate_pool`'s `slot_stride` rounding
   (`ggml_type_size(GGML_TYPE_Q6_K)` = 210, confirm the round-up-to-block
   math and the `s02 = slot_stride_bytes / ts0` blocks-per-slot calculation
   in `ggml_cuda_moe_cache_mmv` both come out exact, not off-by-a-partial-
   block, for a 210-byte block size specifically.
2. Instrument `moe_cache_dispatch`/`moe_cache_collect` specifically for
   `ffn_down_exps` calls (or generically, print the `wtype`/`slot`/`pin`
   details of the SPECIFIC row that goes non-finite) under this exact
   2-prompt CLI reproducer - it's now cheap and fast to iterate on (single
   process, sub-second to reach the failure, no model reload needed for
   re-runs within the same process if the harness is extended to loop).
3. Check whether the SAME set of slots/experts admitted during prompt 1's
   `ffn_down_exps` dispatches are the ones read (cache HIT) during prompt
   2's first `ffn_down_exps` call - i.e. whether this is specifically an
   eviction/re-admission-driven slot, or a slot that was resident and
   untouched since prompt 1 requesting a hit on already-stable data (the
   two point at very different mechanisms: a slot mutating in a way none of
   this session's or the prior session's checks catch, versus something
   wrong in the admission path itself for this specific tensor/type).
4. The `examples/eval-callback` two-prompt harness (kept in the tree, real
   debugging capability) is currently hardcoded to a single greedy-decode
   loop shape per prompt and exits on the first NaN - extending it to loop
   N prompt-pairs without re-loading the model would make bisecting #1-3
   above much faster than restarting the process each time.

**Also surfaced, not yet investigated**: the `llama_memory_recurrent`
wildcard-seq_rm assertion failure above is architecture-specific (hybrid
SSM models) and separate from the moe-cache bug, but is itself a real crash
on a documented, legal API call (`llama_memory_seq_rm(mem, -1, p0, p1)` is
specified to match any sequence) - worth its own investigation and fix,
just not this session's focus.

## Attempted fixes, all tested against the fast CLI reproducer, all failed - and a major correction

User asked to fix the bug rather than keep narrowing it. Given the
"admission alone, no dispatch needed" evidence, tried closing the window by
construction:

- **`cudaDeviceSynchronize()` wrapped around the fill worker's H2D copy**
  (before AND after, in `moe_cache_worker`, `moe-cache.cu`) - the direct
  test of "this fill's real device traffic races something else running
  concurrently on the main thread's own stream(s)." Rebuilt, re-ran the
  2-prompt reproducer: **still corrupts**. Reverted (kept as a documented
  negative result, not left in as dead-weight latency).
- **`GGML_CUDA_MOE_CACHE_NO_RT_ALLOC=1`** (preallocates the fill worker's
  stream and pinned staging buffer once at startup instead of lazily per-job
  - rules out a stream-creation or `cudaMallocHost`/`cudaFreeHost`
  reallocation mid-session as the timing culprit): **still corrupts**.
- **`--moe-cache 64`** (much smaller budget, far less VRAM claimed than the
  512 used throughout this investigation - a direct test of "moe-cache's
  VRAM footprint shifts where some OTHER buffer gets placed, exposing a
  latent bug elsewhere"): **still corrupts**.
- **`GGML_CUDA_MOE_CACHE_NO_HOST_PTR=1`** (disables the separate CPU-side
  "read from a cache-owned host copy instead of `src0->data` directly"
  redirect in `ggml-cpu.c`, gated on `ggml_moe_cache.host_ptr` returning
  non-null): **still corrupts**.

**Major correction while investigating the fixes**: checked with
`MOE_CACHE_DEBUG_GATE=1` whether `moe_cache_begin` is even CALLED for
`blk.0.ffn_down_exps` on the second prompt's PREFILL (the call that
actually goes non-finite, per the eval-callback trace) - **it is not**.
It's called 40 times during prompt 1's greedy generation (once per
decoded token, confirming this exact tensor genuinely is moe-cache-managed
in general), but zero times for prompt 2's prefill step. The eligibility
gate in `ggml-cpu.c` (`n_ids * ids->ne[1] <= MOE_CACHE_MAX_TOPK` and
`moe_cache_max_tokens_ok(ids->ne[1])`) bypasses moe-cache entirely for
any multi-token batch over its threshold - prefill of a ~52-59 token
prompt in one ubatch crosses it, so **moe-cache's own begin/plan/dispatch/
collect pipeline, its circuit breaker, and every one of this session's and
the prior session's consistency checks are ALL simply not in the call
path for the specific op that goes bad.** This retroactively explains why
`OUTPUT REJECTED` never once fires for layer 0 in the full-diagnostic
server run above despite the corruption being real and present there too.
`NO_HOST_PTR` ruled out the one OTHER moe-cache touchpoint that runs
regardless of the node/pipeline (the CPU-side host-copy redirect), so the
actual failing computation, for this specific manifestation, is
**completely standard, unmodified ggml `mul_mat_id` code - no moe-cache
logic executes for it at all.**

This means the defect is not "in" moe-cache's dispatch, admission, or
bookkeeping code as such - it requires moe-cache to be ENABLED (config on
vs off is the only lever that reliably separates clean from corrupt) but
manifests through a call that never touches moe-cache's own code. The most
coherent remaining explanation: moe-cache's mere presence - its VRAM
allocation, its hooks into `ggml_backend_sched` (session take/adopt across
reserves), or something about how it changes the graph/scheduler's
behavior even for ops it declines to manage - perturbs something in a
SEPARATE subsystem enough to expose a latent bug there. This model
(`Ornith-1.5-35B`) is a hybrid SSM/attention/MoE architecture with its own
`llama_memory_recurrent` state (`cache_r_l0`/`cache_s_l0` conv/SSM state,
`GATED_DELTA_NET`/`SSM_CONV` ops) - a `llama_memory_recurrent` assertion
crash was already found this session on a different code path (the
wildcard `seq_rm`), making this subsystem a live suspect independent of
that specific crash. Not yet tested: whether the same 2-prompt reproducer
corrupts on a PLAIN (non-hybrid, non-SSM) MoE model with moe-cache enabled
- if it doesn't, that would confirm the hybrid architecture itself is a
necessary ingredient, not just this specific model being unlucky.

A `compute-sanitizer --tool memcheck` run of this exact fast CLI reproducer
was launched to look for a genuine out-of-bounds read/write directly,
rather than continuing to guess at synchronization fixes - see the next
entry for its result once it completes.

## Cross-architecture test and first memcheck run

**The 2-prompt CLI reproducer also corrupts `gemma-4-26B-A4B` (plain,
non-hybrid, non-SSM transformer MoE, `--moe-cache 512`, no `-ncmoe`).**
Rules out the hybrid SSM/GatedDeltaNet architecture as a necessary
ingredient - this is a general moe-cache defect, not something specific to
`Ornith`'s recurrent-memory subsystem. (Matches the much earlier session
note that gemma-4 degenerates identically to Ornith, just emitting a
different repeated token - now confirmed on the SAME fast, precise
reproducer instead of the old slow one.)

**First `compute-sanitizer --tool memcheck` run of the fast CLI reproducer**:
706 errors - but every one of them is `cudaErrorNotSupported` on
`cudaHostRegister`/`cudaGetLastError`, all originating from
`ggml_backend_cuda_register_host_buffer` during MODEL LOADING
(`llama_model_loader::load_all_data`), not during the 2-prompt generation
window the bug lives in. This is `cudaHostRegister`'s well-known
incompatibility with compute-sanitizer, not a real bug - `-ncmoe`
CPU-resident expert regions get host-registered for faster H2D transfer
regardless of moe-cache (a completely separate ggml-cuda mechanism,
`GGML_CUDA_NO_PINNED` to disable, unrelated to moe-cache's own
`hostreg_mb`/`GGML_CUDA_MOE_CACHE_HOSTREG_MB` which stayed at its default of
0 throughout - the earlier "cudaHostRegister" log lines seen in normal runs
of this session were from THIS mechanism, not moe-cache's, a correction to
earlier framing in this doc). Re-running with `GGML_CUDA_NO_PINNED=1` to get
a clean signal for the actual bug window - result below once it completes.

**Second run, `GGML_CUDA_NO_PINNED=1`**: identical result, 706 errors, same
breakdown - the env var didn't change anything (either not read fresh at
the point that matters, or this specific `-ncmoe` host-registration path
isn't gated by it the way assumed). Not chased further given the errors are
unambiguously all from model loading regardless (same stack trace, same
count, both runs) - genuinely orthogonal to the 2-prompt generation window
the bug lives in.

**Conclusion: compute-sanitizer's memcheck found ZERO memory-safety
violations (no out-of-bounds read/write, no uninitialized read) during the
actual bug window, across two full runs.** Combined with everything else
ruled out this session (weight-byte mutation, pin/generation races, dispatch
bookkeeping, the host_ptr redirect, fill-worker/main-thread device
synchronization, runtime allocation timing, cache budget size, the SSM/
hybrid architecture), this now looks less like a classic memory-safety bug
memcheck's instrumentation is built to catch, and more like either (a) a
pure logic defect - correct, in-bounds memory access, but the WRONG value
used or computed (e.g. a stale-but-valid pointer, a size/shape mix-up that
still lands in-bounds) - or (b) a genuine host-side race (a plain C++ data
race between the fill worker thread and the main thread, which memcheck's
GPU-memory-focused checks would not catch at all, unlike TSan). This
session already tried the two most obvious "add more synchronization"
mitigations (full device sync around the fill copy; eliminating runtime
allocation) without effect, which argues against a simple GPU-stream-timing
race specifically, but does not rule out a host-side data race outside
GPU-memory instrumentation's view.

Given the extent of this session's rule-outs, root cause is not pinned down
to a specific fixable line. Asked the user directly how to proceed rather
than shipping a fix that hasn't been verified to actually close the
reproducer.

## Ablation series: every specific admission/hit sub-mechanism individually ruled out

User asked to keep investigating. Also checked the leloch/moe-cache* remote
branches (this session's moe-cache.cu descends from `leloch/moe-cache`,
itself a rename of an independent "EC3" rewrite) for a matching bug -
`EC3_READINESS.md` documents two real, confirmed "silent numerical
corruption" bugs, but both are specific to a GLU gate/up fusion optimization
our code doesn't have at all (no analogous construct exists to patch); the
other findings (OOM-abort hygiene, pool-ordering race) are either already
handled differently in our code or not reachable in a single-model scenario.
No transplantable fix came out of this - recorded for completeness, not
pursued further.

Went back to direct ablation on the fast CLI reproducer, using the
env-var fault-injection points already built (`GGML_CUDA_MOE_CACHE_FAIL`)
plus new ones added this session, individually disabling each specific
mechanism in the admission/hit path to find which one is *necessary*:

| Config | Result |
|---|---|
| `FAIL=insert` (admission entirely blocked, slot never leaves `copying`) | **clean** |
| `FAIL=dispatch` (admission completes normally; GPU kernel never runs, CPU fallback used) | corrupts |
| `FAIL=dispatch` + `SKIP_COPY` (now fixed to actually take effect on the real, default stage-buffer path, not just the never-taken hostreg path - see below) - admission bookkeeping completes, **zero bytes ever copied to the device slab** | corrupts |
| + `GGML_CUDA_MOE_CACHE_NO_SEGMENT_PUSH=1` (new) - slot marked `valid` but never linked into the probation/protected eviction list | corrupts |
| `GGML_CUDA_MOE_CACHE_NO_HITS=1` (pre-existing) - hit accounting (heat/promotion/pin) runs in full, but the hit is never reported to `ggml-cpu.c`, so that row takes the 100% ordinary compute path | corrupts |
| + `GGML_CUDA_MOE_CACHE_NO_PROMOTE=1` (new) - on top of NO_HITS's full accounting, also skip `moe_cache_promote_to_protected` specifically, keeping `readers++`/heat/pins | corrupts |
| `MOE_CACHE_ROW_AUDIT=1` during a corrupting run | **0 mismatches** - row accounting (matrix_rows/matrix_row_counts, including the dispatch-failure fallback path in `ggml-cpu.c`) provably balances even while corrupting |

**Also found and fixed a real bug in this session's own instrumentation
while doing this**: `GGML_CUDA_MOE_CACHE_SKIP_COPY=1` only ever guarded the
`direct` (hostreg) copy branch - which fails to activate by default anyway
(`cudaHostRegister` returns "already mapped", colliding with `-ncmoe`'s own
unrelated host-buffer registration), so the flag silently had NO effect on
any default-configuration run, including whenever it was referenced earlier
in this investigation or the prior session. Fixed to guard all three copy
branches uniformly (`moe-cache.cu`, the fill worker's copy dispatch).

**Net result: every individually-testable sub-mechanism inside "admission
reaches completion" - the real byte copy, the GPU dispatch/kernel, the
eviction-list linkage, hit-promotion/segment demotion, and hit-row
reporting to the CPU compute path - has now been ruled out on its own.**
What's left common to every corrupting configuration and absent from the
one clean configuration (`FAIL=insert`) is only: `slot.state` transitioning
to `valid`, `device->demand_count.erase()`, `device->fills++`, and (on the
hit side) `slot.readers++`/`slot.heat` incrementing and a pin being recorded
- all trivial scalar/map operations that don't plausibly corrupt a tensor's
values on their own, and none of which touch GPU memory or any tensor data
at all. This pattern - many specific mechanisms ruled out, only trivial
bookkeeping ops common to every failure - now reads less like "a specific
line has a bug" and more like **admission reaching completion changes
something incidental (timing, scheduling, or simply that non-trivial CPU
work now happens on this thread at this point) that exposes a bug
elsewhere entirely**, most plausibly in attention/KV-cache computation for
the SAME token, given `NAN_PROBE` already showed the earliest-detectable
NaN is in the activation feeding an early MoE layer - upstream of anything
moe-cache itself computes, in every run this session pinned down a specific
tensor for. Not yet tested: whether admission's CPU cost alone (no cache
logic at all - just busy-spinning the fill worker thread for a comparable
duration) reproduces it, which would be the direct test of "timing/
scheduling perturbation" as the actual mechanism rather than anything in
moe-cache's own code.

**Follow-up: ThreadSanitizer run of the same reproducer** (a pre-existing
`build-tsan` tree from the prior session, CUDA-enabled; built
`llama-eval-callback` into it and ran the identical 2-prompt test). TSan
instruments host/CPU code only, not GPU kernels, so it's the direct check
for a host-side data race that memcheck's GPU-focused checks can't see -
the leading remaining hypothesis after two clean memcheck runs.

**137 warnings, none new or moe-cache-specific.** Triaged by stack trace:

- **1 `moe_cache_plan` double-lock report** (`moe-cache.cu:5736`) - the
  EXACT SAME finding already investigated and closed out in the prior
  session (commit `933f8b625`, "close the TSan double-lock report - not a
  real reentrancy bug"). Not new.
- **136 generic `ggml`/`llama` infrastructure races**, zero of which
  mention `moe_cache`/`moe-cache` anywhere in either thread's stack.
  Overwhelmingly dominated by one pattern: `ggml_new_tensor_impl`
  (`ggml.c:1808`, graph-build-time tensor metadata creation) racing
  against a graph-COMPUTE-time read from a worker thread
  (`ggml_compute_forward_mul_mat_id`, `quantize_row_q8_K_ref`,
  `ggml_compute_forward_swiglu`, `dequantize_row_q4_K`,
  `llama_kv_cache::set_input_kq_mask`, etc.) - i.e. ggml's own
  build-then-compute handoff via its custom spin-wait threadpool barrier,
  which `ggml_barrier()` already carries an explicit, deliberate TSan
  accommodation for (`ggml-cpu.c`: "TSAN doesn't support standalone fence
  yet, we use a dummy read-modify-write instead") - evidence this class of
  false positive is known and only partially worked around, not evidence
  of a real bug in each instance. None of these races involve anything
  this investigation has touched (moe-cache, mmid.cu, mmq.cu, mmvf.cu,
  ggml-backend.cpp), and the SAME pattern would presumably appear on ANY
  ggml threadpool workload, cache-enabled or not - not selective for the
  corrupting condition the way a real culprit would need to be.

**Net result: two full compute-sanitizer memcheck runs (raw GPU memory
safety) and one full ThreadSanitizer run (host-side data races) all came
back clean of anything new or moe-cache-specific**, on the exact
2-different-prompts CLI reproducer that deterministically corrupts the
non-instrumented build every time. Combined with every other rule-out this
session (weight-byte mutation, pin/generation races, dispatch bookkeeping,
the host_ptr redirect, fill-worker/main-thread device synchronization,
runtime allocation timing, cache budget size, non-hybrid-architecture
generality via gemma-4), this investigation has now exhausted the tooling
available to localize the defect mechanically. The corruption is real,
deterministic, and moe-cache-dependent (`--moe-cache off` stays clean on
every tested config) - but does not present as a memory-safety bug or a
detectable data race under either sanitizer, on the fastest and most
precise reproducer found across two full investigation sessions.

**The single most important new finding: a plain CLI decode loop does NOT
reproduce it.** Extended `examples/eval-callback` (kept in the tree - a
genuinely useful capability it was missing) to (a) construct
`common_debug_cb_user_data` with `abort_on_nan=true`, aborting at the first
tensor anywhere in the graph whose value sums to NaN, printing every tensor
name/op up to that point, and (b) actually generate tokens (greedy-decode
loop) rather than the example's original single prefill-only `llama_decode`
call - prefill alone was verified completely clean through every layer,
0-39, output projection included, so the bug needs real decode-step
generation to reach at all. Ran the IDENTICAL model, `-ngl 99 -c 4096
--moe-cache 512 -ncmoe 30`, and prompt, for 40 greedy decode steps under
this instrumentation: **zero NaN, zero abort, clean generation the whole
way** (tokens decode to a plausible step-by-step answer). The exact same
config that corrupts on request 1 of `llama-server`, every time, does not
corrupt at all under a raw single-sequence CLI decode loop.

**What this means**: the bug is not in moe-cache's data (weights, slots,
pins - all directly verified clean) and not reachable via a bare
`llama_decode` loop with the same cache config. It requires something
`llama-server` specifically does: continuous-batching slot/KV-cache
management, CUDA graph capture/replay (the server logs report thousands of
"graphs reused" - `eval-callback`'s per-tensor `ggml_backend_tensor_get`
readback almost certainly forces synchronous, non-graph execution, which
would also incidentally fully serialize every op against the moe-cache
background fill worker's own stream - a difference that could easily explain
why a genuine race would vanish under it), or the request-warmup pass every
server does before serving real traffic. This narrows the search from "the
whole moe-cache subsystem" to "whatever `llama-server` does differently from
a bare decode loop while a cache is active," which is a much smaller and
more promising surface than anything mapped out so far. Not yet identified
which specific difference is responsible - next step is bisecting the
server's own decode path against the CLI's (starting with: does the server's
warmup pass alone, with zero real requests after, already leave something in
a bad state; and whether disabling CUDA graphs specifically in the SERVER
context reproduces the eval-callback's clean result).

(Historical note on the eval-callback finding above: the observation itself
was correct - a bare CLI decode loop really doesn't reproduce it - but the
inference drawn from it, that the bug required something specific to
`llama-server`, was a red herring. The eval-callback tool's per-tensor
readback (`ggml_backend_tensor_get` after every op, for the NaN-abort check)
forces enough synchronous host/device round-tripping that it happens not to
exercise the actual defective code path the same way. The real mechanism,
below, fires unconditionally inside `ggml-backend.cpp`'s scheduler and has
nothing to do with continuous batching, CUDA graphs, or server warmup.)

---

## ROOT CAUSE FOUND: the `////...` corruption, fully closed

### Summary

The corruption was a real, deterministic logic bug in
`ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp`) - not a
memory-safety violation, not a data race, and not specific to moe-cache's
own admission/eviction/pin logic (all of which were correctly exonerated
over the course of this investigation). It required moe-cache's LFRU
device-to-device (D2D) cache-hit path, added in commit `366fe539c`, to be
genuinely active - which on this Ornith-1.5-35B model required an earlier
fix (`a2d412e9`, lowering `min_expert_bytes` so Ornith's 840 KiB experts
stop being silently rejected by a 1 MiB floor). Every clean result recorded
earlier in this document at commits before `a2d412e9` was clean because the
cache was **silently inert** on this model (zero experts ever registered),
not because the code was correct - that inertness is itself worth fixing
separately, since it can silently defeat the whole feature on any small-expert
architecture without the operator ever finding out (the `a2d412e9` warning
log addresses the "find out" half of that, at least).

### The mechanism

`ggml_backend_sched_compute_splits` uploads each MoE weight tensor
(`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) to the GPU by copying only
the rows (experts) actually selected by that token batch's routing decision
- a pre-existing optimization, not something this session added. To avoid
re-decoding the routing tensor (`ids_tensor`, i.e. `selected_experts`) for
every one of the three weight tensors in a layer, the function caches the
decoded "which experts are used" bitmask (`used_ids`) and only recomputes it
when `ids_tensor` changes - reasonable, since `gate_exps`, `up_exps` and
`down_exps` all share the exact same `selected_experts` tensor object for a
given layer/batch (see `llm_graph_context::build_moe_ffn` in
`src/llama-graph.cpp`, which passes one `selected_experts` tensor into all
three `build_lora_mm_id` calls).

Commit `366fe539c` ("moe-cache: LFRU hit/miss D2D split for prefill...")
added a genuine optimization on top of this: before falling back to a
host-to-device (H2D) PCIe copy for a used expert row, check whether that
expert is already resident in moe-cache's decode-time GPU pool and, if so,
serve it via a cheap device-to-device (D2D) copy instead, clearing that
expert's bit in `used_ids` so the H2D copy below skips it. This scan-and-clear
was placed *inside* the same `if (ids_tensor != prev_ids_tensor)` block as
the routing decode - i.e. it only runs when the ids tensor actually changes.

That's the bug. LFRU pool residency is tracked **per weight tensor**
(keyed by `(host_base, expert)`, where `host_base` is the specific tensor's
own host pointer - `gate_exps`, `up_exps` and `down_exps` are three
different arrays with three different `host_base` values and independently
tracked cache contents). But `used_ids` and its LFRU-cleared bits were being
reused **across** `gate_exps`, `up_exps` and `down_exps`, because all three
share the same `ids_tensor` and the recompute (including the LFRU scan) was
gated on `ids_tensor` alone. Concretely: if expert 5 happens to be
D2D-cache-resident for `gate_exps` but not for `up_exps`, the code clears
bit 5 while processing `gate_exps`, then - because `ids_tensor` hasn't
changed - skips the LFRU rescan entirely while processing `up_exps` and just
reuses the same, now-stale, bitmask. Expert 5's row in `up_exps`'s staging
buffer is left with **neither** a D2D copy (the rescan that would have found
it wasn't in `up_exps`'s pool never ran) **nor** an H2D copy (its bit was
already cleared) - the GPU buffer for that row is whatever was there before,
fed directly into `MUL_MAT_ID` as expert weight data. That's why the observed
symptom was NaN/garbage activations at a specific layer rather than zeros:
uninitialized/stale device memory, not a cleanly zeroed buffer.

This also explains every property established earlier in this
investigation:
- **Needs moe-cache genuinely active + populated**: the LFRU D2D path is the
  entire mechanism: no D2D hits, no stale-bit reuse, no bug.
- **Needs the second of two sequential different-content prompts**: the bug
  fires when the *set of experts D2D-hit-resident* differs between
  `gate_exps` and `up_exps`/`down_exps` for the same routing decision -
  which routing-dependent cache occupancy makes far more likely once the
  cache has real, non-uniform history behind it (i.e. after a first prompt
  has already run and populated the pool unevenly across the three tensors).
- **Not architecture-specific**: this is generic MoE scheduling code in
  `ggml-backend.cpp`, unrelated to Ornith's hybrid SSM design - matches the
  earlier cross-architecture confirmation with gemma-4-26B-A4B.
- **Clean under both sanitizers**: there is no unsynchronized memory access
  here at all - every copy that *does* happen is correctly ordered and
  pinned. The bug is a pure logic error (a copy that should have happened,
  silently doesn't), which neither compute-sanitizer nor TSan has any way to
  flag.
- **Independent of cache size/eviction**: the conflation happens regardless
  of how large the pool is or how often eviction runs - it only depends on
  *which* tensor's residency state was scanned last for a shared ids_tensor.
- **Not reproduced by the `eval-callback` CLI loop**: unrelated to this
  mechanism - eval-callback's forced per-tensor synchronous readback happens
  to interact with the scheduler's split/copy pipeline differently. Not
  investigated further since the actual root cause was found by this point.

### Bisection results (101-commit range, `64aa6a77c`..`5de4ff917`)

| Position | Commit | Cache state on this model | Result |
|---|---|---|---|
| 0 | `64aa6a77c` | inert (min_expert_bytes=1MiB rejects Ornith's 840KiB experts) | CLEAN (uninformative - cache did nothing) |
| 39 | `2a36936a6` | inert | CLEAN (uninformative) |
| 47 | `2f41751d9` | inert | CLEAN (uninformative) |
| 52 | `085ddad77` | inert | CLEAN (uninformative) |
| **54** | **`a2d412e9`** | **genuinely active** (fixes min_expert_bytes 1MiB→256KiB) | **CLEAN** (first valid data point - cache works correctly here) |
| **55** | **`366fe539c`** | genuinely active | **CORRUPT** (`////...`, uniq-char-ratio 0.01) - **introducing commit** |
| 57 | `140869276` | genuinely active | CORRUPT |
| 76 | `fcbee0e12` | genuinely active | CORRUPT |
| 101 | `5de4ff917` | genuinely active | CORRUPT |

The flip is exactly at commit `366fe539c2976c4cfe59efce248fa08d335bb861`,
matching the mechanism identified by direct code reading above. Positions
0/39/47/52 were re-examined and found to be false-clean (cache silently
inert, not correctly-functioning) - only 54 onward are meaningful data
points, and the flip from 54 (clean, active) to 55 (corrupt, active) is a
real signal, not an artifact of cache inertness.

### The fix

Applied to `ggml/src/ggml-backend.cpp`, `ggml_backend_sched_compute_splits`
(current code, evolved from the single-expert `moe_lfru_copy_expert` at the
introducing commit to a batched `moe_lfru_copy_experts` API by the time this
reached the tip of this branch - same underlying bug, same fix location).

Separated the two conflated concerns:
- `used_ids`: pure routing decode ("which experts does this ids_tensor
  select"), legitimately shared across `gate_exps`/`up_exps`/`down_exps`
  since they share the same `ids_tensor` - kept exactly as before, still
  only recomputed when `ids_tensor` changes.
- `copy_ids` (new): a fresh copy of `used_ids`, taken and re-scanned against
  the LFRU cache for **every input tensor**, not just when `ids_tensor`
  changes - because LFRU residency is per-weight-tensor and must be
  rechecked independently for `gate_exps`, `up_exps`, and `down_exps` even
  when they share a routing decision. The LFRU scan-and-clear,
  `any_h2d_needed` computation, and the downstream contiguous-run H2D copy
  loop all now operate on `copy_ids` instead of mutating `used_ids` in
  place.

This preserves the original optimization intent (routing is decoded once
per `ids_tensor`, not once per weight tensor) while making the D2D-hit
bookkeeping correctly scoped per weight tensor, which is what it actually
needs to be.

### Verification

Rebuilt `llama-server` on the main working tree with the fix applied and
re-ran the standard reproducer (two sequential different-content prompts,
`-ngl 99 -c 4096 -ncmoe 30`, moe-cache genuinely active - confirmed via the
`[moe-cache] first fill` log line):

- Original 2-prompt reproducer (pencils / garden-area): clean, correct
  arithmetic in both responses.
- Two longer (n_predict 250), topically unrelated follow-ups (a short story;
  a TCP vs UDP explanation) run sequentially after the first two: both fully
  coherent, on-topic, no repeated-character corruption. (Note: the uniq-char-
  ratio heuristic used throughout this investigation is not well-calibrated
  for longer structured text - lower ratios here reflect normal prose/list
  repetition, not corruption; verified by reading the actual response text.)
- Three additional short sequential math prompts with varied numbers: all
  arithmetically correct (117/3=39, 134/4=33.5, 151/5=30.2).

No corruption observed across any of these varied, sequential, real-routing
requests on the fixed build. The fix is applied but **not committed** -
left as an uncommitted change in the working tree pending review.

## Atlas warming, second pass (2026-08-29): from actively harmful to safe, still short of a win

Item #9 from the pending-work queue ("integrate a mechanism that actually
CONSUMES the atlas") revisited with a proper multi-round, multi-switch
benchmark, since the only prior evidence for Track 1's atlas warming
(`docs/plan.md`, Track 1 step 3 and its follow-ons) was a single-pivot test
and a steady-state one - neither matches real dynamic usage. New harness:
`atlas_topic_switch_bench.py` - 6 sequential topic pivots drawn from the
atlas's own measured categories (code -> medicine -> law -> creative ->
science -> history), 4 rounds per arm, hit rate measured per-request via
`/experts`' hit/miss counters (not cumulative session average - the delta
across each individual request, isolating the post-switch effect the
original single-pivot test measured). All four numbers below are the
post-switch (requests 1-5, excluding the non-switch first request) pooled
hit-rate delta against a matching no-warming baseline, same methodology
throughout so they are directly comparable.

**1. Thin eviction (the shipped design going into this session, admission's
own eviction capped at 1/pass): -0.31pp.** Warm never won requests 0, 1, or
4 across all 4 rounds - a small but real, consistent regression, not noise.

**2. Wide eviction ("burst mode" - shorter check interval, bigger
admit/evict budget, multi-tensor sweep on a detected topic shift, admit's
own eviction cap widened to 6/pass during a burst): -6.55pp - six times
worse, warm losing on literally every single request across all 4 rounds
(0/4 wins everywhere).** This was the pivotal result. The user diagnosed
the mechanism from first principles before this number existed: atlas
admission's eviction path sacrifices a currently-resident, reactively-
admitted expert to make room for a *predicted* one, and unlike the reactive
path's own eviction (always justified - it satisfies a request that just
actually happened), a wrong prediction is a real, direct loss. Widening the
eviction budget only gave that mechanism more chances to make a bad trade,
which is exactly what the data shows.

**Root cause, confirmed in code, not just data**: `moe_cache_slot_reset`
takes an `add_to_free` flag. The reactive/demand-driven admission path
(normal cache misses) calls it with `add_to_free=false` on every eviction -
the freed slot is reused immediately in place, never returned to
`pool.free_slots`. That list is populated once at pool creation and
essentially never again during normal operation. Atlas admission's own
"free-slot-only" fallback (the pre-eviction-fix original design, and the
first fix attempted this pass) therefore has almost no room to ever act
after the first few tokens of any real run - which is exactly why it had
previously measured "flat, no effect either way" and why eviction rights
were added to it in the first place. Reverting to free-slot-only alone
would trade "actively harmful" for "provably safe but likely near-inert" -
correct, but not the fix that delivers a real win.

**3. Free-slot-only, no eviction at all: not separately measured.** The
code-level reasoning above was trusted rather than spending another full
benchmark cycle confirming a predictable near-zero result; moved directly
to the actual fix instead. (An `atlas_bench_results_FREESLOT_ONLY.json`
file exists in the scratch directory from this pass, but it is a
leftover duplicate of the wide-eviction result, not a genuine measurement -
noted here so it is not mistaken for real data later.)

**4. The actual fix - separate "whose eviction" from "eviction at all"**:
the reactive path evicts constantly anyway, for real reasons. The
principled integration point (already identified but abandoned once
before, in a much earlier pass of this same investigation - see the
"eviction-weighting" removal note still in the code history) is to let the
atlas influence *which* slot the reactive path sacrifices when it was
evicting regardless, never to trigger a new eviction event on its own. The
earlier attempt at this was reverted specifically because it interfered
with atlas admission's *own* eviction rights ("admission ends up fighting
itself") - a confound that no longer exists once admission's own eviction
is removed entirely. Implemented as `moe_cache_atlas_align_score` (the same
cosine-similarity-times-specialization score `moe_cache_atlas_rank` already
computes for admission candidates, evaluated against whatever is currently
resident instead), folded into `moe_cache_weighted_heat` as a strictly
boost-only multiplier: `heat *= 1 + align_score * strength` where
`align_score` is clamped to >= 0. A resident slot's effective heat can only
go *up* when it is topically aligned with the live request direction, never
down for one that is not - unmeasured or misaligned slots score exactly as
they always did. This cannot make eviction target selection worse than
baseline by construction, only ever protect a slot that would otherwise
have been picked.

Combined with everything kept from the burst-mode work (wider admission
candidate pool, burst-triggered check frequency, multi-tensor sweep - none
of which can hurt either, since they only mean more chances to fill a
genuinely free slot, never a chance to take one away) and a separate fix
found along the way (the main VRAM fill path - both reactive and
atlas-triggered admissions - never released an expert's now-redundant mmap'd
host pages after copying it to VRAM; only a different, off-by-default
host-hot-buffer mechanism had the equivalent call. Added
`moe_cache_release_source_pages` to the shared fill-completion path,
provably safe to call unconditionally there since the copy that read those
bytes has already completed and synced by that point).

**Result, same benchmark, same methodology: post-switch pooled delta
-0.35pp, warm won 0/4 rounds on the pooled per-round average - but the
per-round numbers are now close (off: 0.8039/0.8041/0.8041/0.8030, warm:
0.7973/0.8017/0.7991/0.8028 - round 4 is within noise of tied), and
individual requests are mixed rather than uniformly losing: law (request 2)
is a real win, 3/4 rounds, +0.40pp; the first request (request 0, not
actually a topic switch - included as a sanity check, not part of the
post-switch average) shows a striking -9.09pp, 0/4, worth investigating
separately since it may indicate the protection boost is acting on noisy,
not-yet-stabilized `req_dir` readings very early in a session before the
centroid has had time to settle.**

**Honest verdict**: the eviction-weighting redesign achieves the safety
property it was built for - no longer actively harmful, unlike both prior
designs this session (-0.31pp and -6.55pp) - but does not yet deliver the
consistent positive win the 95%+ hit-rate goal needs. It has landed at
roughly the same small-negative magnitude as the very first, simplest
design tested, just via a mechanism now understood to be safe rather than
one proven destructive. Not shipped as default. Plausible next steps, none
attempted yet: tune `GGML_CUDA_MOE_CACHE_ATLAS_ALIGN_PROTECT_STRENGTH`
(default 3.0) higher, since the current boost may simply be too weak to
meaningfully shift real eviction outcomes; investigate the request-0
anomaly specifically, since a fix there might flip more than just that one
request; or accept that a purely protective, no-new-churn mechanism has an
intrinsically low ceiling and pair it with something that can proactively
seed new content without eviction risk (e.g. the cross-run frequency-
persistence idea sketched but not built this session, which needs no
eviction at all - it only competes for the free-slot budget already used
during the natural startup window).

**Shipped mechanism and its full knob set** (all in
`ggml/src/ggml-cuda/moe-cache.cu`), for reproducibility:
- `GGML_CUDA_MOE_CACHE_ATLAS_WARM` (off by default) - the master switch.
- `GGML_CUDA_MOE_CACHE_ATLAS_WARM_RANK_K` / `_ADMIT_K` (steady-state
  candidate pool / admit count, default 4 / 2) and their
  `_BURST_RANK_K` / `_BURST_ADMIT_K` counterparts (default 24 / 12).
- `GGML_CUDA_MOE_CACHE_ATLAS_WARM_INTERVAL` / `_BURST_INTERVAL` (plan()
  calls between throttle checks, default 64 / 4).
- `GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_THRESHOLD` (req_dir movement
  distance that counts as a detected shift, default 0.25).
- `GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_TENSORS` (distinct tensors swept
  per burst firing, default 8).
- `GGML_CUDA_MOE_CACHE_ATLAS_ALIGN_PROTECT` (on by default once
  `ATLAS_WARM` is on) / `_STRENGTH` (default 3.0) - the eviction-weighting
  boost.
- `GGML_CUDA_MOE_CACHE_RELEASE_FILL_SOURCE` (on by default, independent of
  atlas warming entirely - applies to every fill) - the host-page-release
  fix.
- `GGML_CUDA_MOE_CACHE_ATLAS_WARM_EVICT_MIN` / `_EVICT_CAP` remain in the
  source as dead parameters (the eviction path they configured was
  removed) - kept only so `evict_cap_override`'s call sites did not need
  signature changes; harmless, do not affect behavior.

## Substitution's rank-gated draft/fallback split - VALIDATED, 4/4 rounds (2026-08-29)

Substitution (`GGML_CUDA_MOE_CACHE_SUBSTITUTE`) existed and was measured
(perplexity, router-score) but had never actually been turned on in a live
server this session, and never with the multi-round tok/s methodology
everything else here uses - it doesn't move hit rate at all
(`device.misses` increments unconditionally before the substitution check
even runs), so hit-rate benchmarks were never going to show its effect.

Added a draft/fallback split, gated by the missed expert's own router rank
(`GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK`): this dispatch path has no
access to the router's real probability values (same GPU-only limitation
the "full-probs oracle" comparison elsewhere in this doc already
documents), but rank position is a real, already-measured proxy for a
miss's cost - rank 0 misses 20% of the time vs rank 7's 44%, and ranks 4-7
carry only 32% of the total gate mass. Below the configured floor (the
router's most-confident picks), a miss always pays exact CPU fallback;
at or above it, the cheap resident stand-in serves it - the speculative-
decoding "draft, verify by cost estimate, fall back" shape, using rank as
the cost estimate since real probabilities aren't reachable here.

**Result, multi-round topic-switch benchmark (tok/s, not hit rate - see
above for why), temperature 0, 4 rounds/arm, `MIN_RANK=4`:**

| | off | substitute | delta |
|---|---|---|---|
| overall | 55.12 tok/s | 59.59 tok/s | **+8.11%** |
| round 1 | 53.11 | 57.09 | +7.50% |
| round 2 | 54.12 | 61.15 | +12.98% |
| round 3 | 56.80 | 60.34 | +6.23% |
| round 4 | 56.43 | 59.78 | +5.92% |

**4/4 rounds won, every one of the six topics individually faster**
(+1.47% to +16.08%), hit rate confirmed unchanged (0.766 vs 0.767 in the
correctness pass) - this is a clean, consistent, multi-round-validated
speed win, not a single noisy sample. Quality cost not independently
re-measured this pass; relies on the existing perplexity bound (+0.47%,
docs/plan.md, "Substitution" row) and the router-score proxy (0.0386 vs a
0.0379 theoretical oracle) - both already argue the router-rank method's
approximation cost is small, but the reasoning-benchmark check (item #5,
one prior run each, 79.2% vs 62.5%, surprising and unvalidated) still
needs more rounds before quality can be called settled, separate from this
speed result.

## Track 4 - "run 320B-class models on this hardware" - SCOPED, not yet built (2026-08-29)

New branch, `track-4` off track-1.6's HEAD (renamed from an earlier `track-2`
that collided with the existing Track 2 heading above). Prompted by asking
whether GLM-5.3-Flash (321B total / 18B active, zai-org, hybrid 34 KDA +
11 MLA/DSA layers, 288 experts/8 routed/1 shared, hidden_size=4096,
moe_intermediate_size=2048 - real config.json values) could hit 40 TPS on
this machine (RTX 3060 12GB VRAM, ~31GiB RAM, WD Blue SN5000 NVMe).

**Verdict on the original ask: no.** Best-case VRAM-bandwidth ceiling with
zero cache misses is ~25-30 tok/s; realistic ceiling given VRAM can only
hold ~12GB against a ~200GB Q4_K_XL model (fast-memory budget doesn't even
cover the 90GB 1-bit floor) is far below that - comparable published
results on FAR more capable hardware (4x3090/4090 + 192GB RAM) land at
3-9 tok/s. A hardware-tier problem, not a software one, for the *whole*
question. Full reasoning and sourcing captured in conversation, not yet
copied into this file verbatim.

**What IS still worth building, independent of hitting 40 TPS on GLM-5.3
specifically** - the queue, in the order scoped so far:

1. **Embedding-row GPU cache.** Real, validated by an independent frontier
   lab at far larger scale (Qwen3.8-Flash-Next / "Qwen4"'s 51B-parameter
   N-gram lookup table, offloaded to host RAM and gathered per-token,
   measured 84GB->60GB per-GPU footprint / 78% more KV headroom / 0.007%
   throughput cost / bit-identical output, independently confirmed by
   SGLang + community builds within days of release). `tok_embd` is
   CPU-resident by default in this fork already (`src/llama-model.cpp:
   1368-1370`) - the win is promoting hot rows into VRAM, not evicting
   them out. Needs new plumbing: `GET_ROWS` has no interception hook
   today (unlike `mul_mat_id`), and it's a shared op (LoRA, MoE routing,
   ALiBi, recurrent state all use it too) so any hook must be tensor-
   scoped to `tok_embd` specifically. Not started.

2. **Heat-aware intra-expert neuron subsetting.** New idea, not published
   anywhere in this exact form (adjacent literature - DejaVu/PowerInfer,
   arXiv:2310.17157 / arXiv:2312.12456 - is per-token-predictive and
   dense-model-only, getting real sparsity out of SwiGLU normally needs
   retraining per ProSparse arXiv:2402.13516; the one MoE-specific paper,
   arXiv:2605.08575, is magnitude-threshold and compute-only, can't touch
   transfer bytes). This fork's version: track accumulated
   `|gate_i * up_i|` per neuron per `(layer, expert)` over time (same
   style of signal as existing LFRU heat, just nested one level deeper),
   and at admission time only transfer/keep the historically-hot neuron
   subset of `gate_proj`/`up_proj` resident - the un-loaded neurons read
   as zero to the (unmodified) `down_proj` matmul, so phase 1 needs no
   new kernel for `down_proj` at all, only smaller loads/matmuls for
   gate/up. `down_proj` itself resists this (its neuron axis is the
   Q4_K quantization-block axis, not row-addressable) and needs a
   one-time repack to a neuron-major layout - deferred to a fast-follow
   phase 2 once phase 1 is proven out.
   Verified against moe-cache's actual architecture before committing to
   this shape (2026-08-29 scoping pass):
   - `ggml_mul_mat_id`'s weight tensor is a single 3D tensor, one shape/
     stride for every expert in a batched call (`ggml/src/ggml.c:
     3329-3353`) - confirmed structurally, not just a kernel-level
     optimization choice, that every expert of a tensor needs the SAME
     resident neuron count per dispatch. A variable per-expert budget
     doesn't fit the op as it exists.
   - moe-cache's own pool design independently hard-codes uniform slot
     size (`moe_cache_pool` has one `expert_size` for the whole pool,
     `moe-cache.cu:161-212`; addressing/guard-byte logic throughout
     assumes it) - per-slot variable sizing isn't structurally possible
     in the current pool type.
   - No per-neuron statistic exists today (`moe_cache_slot::heat` is a
     single `uint32_t`, `moe-cache.cu:156`) - genuinely new state, but
     cheap (~100-200MB total across a realistic layer/expert count).
   - No existing hook observes the actual gate x up intermediate value -
     moe-cache's hooks all anchor to the `mul_mat_id` node itself, which
     fires BEFORE the SiLU/gate-up combine (`llama-graph.cpp:2262-2280`,
     tensors `ffn_moe_silu`/`ffn_moe_swiglu`). Needs a new graph-level
     `ggml_moe_cache_api` callback, the first one not anchored to
     `mul_mat_id`.
   Net: buildable, but a real three-part lift (tiered-K dispatch scheme
   to keep batching legal, a second pool architecture for per-slot
   neuron metadata, a new hook surface) - not a parameter on the
   existing system. Not started.

3. **NVMe read-pattern optimization.** Distinct from both items above -
   not about needing fewer bytes, about reading the needed bytes with
   less latency/overhead (io_uring async reads, larger sequential
   batches, smarter mmap/madvise hints). No GPUDirect Storage in this
   codebase (checked, zero `cuFile`/GDS references in moe-cache.cu).
   Readahead was already tried once this session for a different purpose
   and found to hurt cold-start (~+21% worse, "Readahead re-test" section
   above) - worth keeping in mind as a prior negative result on a
   related lever, though not the same mechanism. Not started, not yet
   scoped in detail.

4. **Weight compression on top of quantization.** Requested, not yet
   scoped. Three genuinely distinct axes here, worth keeping separate
   rather than treating quantization as a form of compression (an
   earlier draft of this note blurred that, corrected 2026-08-29):
   **precision** (quantization - changes the values themselves, lossy,
   but everything stays present at reduced fidelity), **statistical
   redundancy** (this item - generic zstd/lz4-style re-encoding of the
   same bytes, can be fully lossless), and **importance-based omission**
   (item 2 above - not storing some values at all and treating their
   absence as zero, the sparse-matrix/CSR family; lossy like
   quantization, but by omission rather than re-encoding, which is why
   it belongs conceptually with compression rather than as its own
   fourth category). All three compose - quantize, then optionally
   losslessly compress the quantized bytes, independent of which
   neurons get kept at all.
   Expected value for THIS item specifically is real but modest -
   already-quantized bytes have less redundancy left to exploit than
   raw f32/f16 would; likely gets more traction from per-block scale/min
   metadata patterns than from the packed values themselves. Needs a
   prep-time compression pass over the GGUF's expert tensors plus a
   decompression step in moe-cache's fill path (CPU-side before H2D
   transfer is the likely right default, since NVMe bandwidth is the
   established bottleneck here, not CPU headroom). Not started, not yet
   scoped in detail.

**Item 1 (embedding-row GPU cache) - REOPENED, 2026-08-29, scoped to the
320B-class case specifically, not Ornith.** Originally dropped on a sizing
argument that turned out to be measured against the wrong denominator:
"the win is small" was argued relative to the whole ~200GB model, not
relative to what's actually scarce - the ~4-6GB of leftover VRAM
moe-cache's pool has to work with after model/KV/scratch overhead on a
12GB card. Against that number, a ~1GB+ table (Ornith's `token_embd.weight`
measured directly from the gguf: 508M params, ~1GB) is not a rounding
error. Separately worth being precise about: this is not the same
direction of move Qwen3.8-Flash-Next/SGLang validated (they moved a 51B
table OFF VRAM to free capacity; `token_embd.weight` here is already
CPU-resident by default, so the scoped mechanism is promoting hot rows
ONTO VRAM for speed, not freeing footprint) - related but structurally
different, and the "validated at scale" claim should be read with that
caveat, not as a direct precedent.
Explicitly scoped by the user to matter for the 320B-class case
(GLM-5.3-Flash, where every GB of the ~4-6GB scarce cache budget counts)
and NOT for Ornith-1.5-35B, where it isn't needed right now. The mechanism
itself is still real work: picking the hot set is cheap (reuse
`llama-frspec-vocab-trim` + `llm_frspec_load_d2t_sidecar()`, zero new
tooling), but making hot rows actually faster still requires touching
`ggml_compute_forward_get_rows` - generic, shared code every model and
every `GET_ROWS` call in the graph goes through (LoRA, MoE routing, ALiBi,
recurrent state, embeddings). Not started; reopened as a real item for the
320B-class track, not attempted on Ornith.

**Item 2 (heat-aware intra-expert neuron subsetting) - the historical/
reactive-heat framing is suspect, not yet resolved.** Raised independently
during the differential-quantization discussion below and never fully
settled for this item too: a signal based on *past* usage (accumulated
`|gate_i * up_i|` over time) can't tell you a topic shift is about to make
a currently-cold neuron matter, and the damage from guessing wrong here
would be worse than for VRAM residency (irreversible per-request, not
cheaply corrected next token). Whether an atlas/req_dir-informed signal
actually fixes this (as opposed to just moving the same problem one level
down) was never tested or confirmed - genuinely open, not dropped, not
proceeded with either.

**Differential per-expert quantization (shrink cold experts, keep hot ones
full quality) - explored and abandoned, 2026-08-29.** Generated a real
imatrix (Ornith-1.5-35B, single-novel calibration corpus, ~341 chunks) and
started an IQ2_XXS-vs-Q4_K_M full-model requantization to measure a
worst-case quality ceiling. Stopped mid-run, by explicit instruction: "this
we will never be sure of" - the underlying question (how to decide which
experts get the cheap copy, permanently, given the decision can't be
undone the way VRAM eviction can) was never resolved, and the local test
wasn't answering it either. Separately noted: generic quantization-quality
tradeoffs (e.g. IQ2_XXS vs Q4_K_M perplexity deltas) are usually already
published and should be looked up before running a fresh local
measurement - reserve local tests for genuinely session/implementation-
specific claims, not well-established generic facts. Imatrix file left on
disk at /mnt/nvme/models/ornith/shadow/ornith-imatrix.gguf, unused,
harmless. Partial IQ2_XXS output deleted.

**Also confirmed and dropped from scope this pass:**
- CUDA graph capture eviction (originally item 3 of the "three VRAM
  consumers" list) - already done upstream (`81ab64f3c`, `b94050e89`),
  nothing to build.
- Compute/activation scratch buffer right-sizing - re-verified a dead
  end (`ggml_gallocr` already does genuine liveness-based reuse via a
  real free-list allocator, not naive per-node summation,
  `ggml/src/ggml-alloc.c:717-822`) - no sizeable waste to reclaim.
- KV-cache LFRU tiering (the "keep hot pages resident, evict cold ones
  to host RAM" idea, explicitly requested and then explicitly dropped
  once scoped) - attention has no paged-access primitive to hang
  eviction on (dense reduction over the full live KV range every
  decode step, unlike MoE's sparse per-token expert gather), would need
  a new CUDA attention kernel from scratch, and is low-priority for
  this fork's actual target models anyway (MLA/KDA hybrids keep KV cache
  in the low single-digit GiB range at typical context lengths - GLM-5.3
  specifically: 34 of 45 layers use KDA, fixed-size recurrent state,
  independent of context length at all).
- Output head / LM head tiering - explicitly out of scope, dense read
  every step by construction, no sparsity to exploit (same conclusion as
  the original 2026-08-28 scoping note).
- Intra-FFN activation sparsity via a predictive approach (DejaVu/
  PowerInfer-style) - superseded by item 2 above, which uses a
  historical/heat-tracked signal instead specifically to solve the
  transfer-bytes problem the predictive approach can't touch.

## Heat-aware intra-expert neuron subsetting - step 1 built and verified (2026-08-29)

Step 1 of the phased build (gate/up neuron subsetting first, down_proj
repack deferred): the observation hook. Per the earlier verification pass,
no existing moe-cache hook observes the actual gate/up matmul output values
- `.begin`/`.plan`/`.dispatch`/`.collect` all operate on ids and slot
indices, never the computed rows. Turned out not to need a new
graph-level hook at all: `moe_cache_collect()` already downloads and
validates the real matmul output (`device.h_out`) for every gate_exps/
up_exps hit, as part of what it already does - the accumulation only
needed to read data already present there.

**Built:** `device.neuron_heat: unordered_map<moe_cache_key, vector<float>>`
(moe-cache.cu, next to `co_activation`), accumulating `sum(|value|)` per
neuron per `(tensor, expert)` key, gated behind `GGML_CUDA_MOE_CACHE_NEURON_HEAT`
(off by default). Purely observational - nothing reads this map yet.

**Verified, not assumed:**
- No new CUDA kernel, no new sync: confirmed by reading the insertion
  point directly (right after `moe_cache_collect`'s existing `h_out`
  memcpy loop, already inside the `if (ok)` branch after the mandatory
  post-dispatch `cudaStreamSynchronize`).
- Compiles clean (`ggml-cuda` target), no new warnings.
- Actually running, not silently no-op'ing: temporary debug print
  confirmed real collect() calls with `n_out=512` (matches Ornith's
  `expert_feed_forward_length` for gate/up) and `n_out=2048` (matches
  down_exps's hidden width) - removed after confirming, not left in.
- True no-op on inference correctness: same prompt, temperature 0, flag
  off vs on, output diffed byte-for-byte identical - checked twice, on
  two different prompts (photosynthesis, Rome essay), after both the
  debug-print version and the cleaned-up final version.

**Still open, the two harder pieces:**
1. Tiered-K dispatch scheme so `ggml_mul_mat_id`'s batching (one shape per
   call, confirmed structural at `ggml.c:3329-3353`) stays legal when
   different experts have different resident-neuron-subset sizes.
2. The actual admission-time decision (which neurons to keep, atlas/
   req_dir-informed per the corrected design, not raw historical
   accumulation) - `neuron_heat` exists now as a real signal to build that
   on, but nothing consumes it yet.

Not shipped, not benchmarked - this is infrastructure, not a measured
result. No performance or quality claim attached to this step.

## Heat-aware intra-expert neuron subsetting - full measurement pass and a precise, ready-to-execute spec for the remaining mechanism (2026-08-29)

Continuation of the section above. Everything in this section is either
(a) measured, real data from the live `neuron_heat` infrastructure, or
(b) a fully-scoped design for what's NOT built yet - marked clearly which
is which. No performance claim anywhere here; this mechanism has never
been dispatched, only measured and designed.

### What the measurements actually found, in order

**1. Value-threshold concentration (not rank-based - see the correction
below on why rank-based framing was wrong).** For each expert, using ITS
OWN max neuron value as the reference (not a shared/global constant):

| neurons clearing this % of the expert's own max | mass retained | neuron count clearing it |
|---|---|---|
| 50% | ~27-28% | ~18-20% |
| 25% | ~81-82% | ~71-72% |
| 10% | ~99% | ~95-96% |
| 5% | ~99.8% | ~97-98% |

This replaced an earlier rank-based table ("top K% of neurons by count ->
X% of mass") that was the wrong framing entirely - a count-based cutoff
doesn't adapt to each expert's own distribution shape, and using MEAN as
the reference for "dead" was unreliable given how skewed these
distributions are (measured max/mean ratios 120-270x across different
runs). Fixed to: value threshold relative to each expert's own max
(concentration measurement) and each expert's own MEDIAN, not mean (dead-
neuron threshold - median is far more robust to the skew).

**2. Per-expert "required width for 90% coverage", if each expert could
pick its own K freely:** min=6-7%, p25=~80%, median=~83-84%, p75=~85-86%,
p90=~86-87%, max=~88% (n≈14,500-23,500 keys depending on the run).
Bimodal in shape: most experts are genuinely flat (need 80%+ width),
a real minority is sharply concentrated (down to 6-7%).

**3. Per-layer breakdown - INITIALLY looked like "layer 0 is uniquely
special" (median dead-fraction 30.9%, required-width 45.7%, vs ~0%/82-85%
for every other layer, 0-39, confirmed with `-ncmoe 41` for full
visibility). This framing was WRONG, caught on challenge (see below).**

**4. THE CORRECTION - this is not a layer-level effect, it's a per-expert
effect that happens to concentrate in layer 0 but is NOT exclusive to it.**
Checked directly: among 7,598 non-layer-0 experts, 17 individual outliers
(>10% dead-fraction) exist, scattered across layers 1, 2, 3, 4, 37, and 39
- some with very high sample counts (layer 3 expert 163: 23.4% dead-
fraction, mass=141,111 - not a small-sample artifact). The per-layer
MEDIAN masked these because they're a small minority within an otherwise
flat layer. **Correct design: no layer restriction at all - apply the
same per-expert eligibility check (enough accumulated mass, then top-K
selection) to every expert in every layer, and let whichever ones qualify
convert, regardless of which layer they're in.** This is simpler code,
not harder - removing an artificial gate, not adding one.

**5. Throughput signal (per-layer miss rate) - deliberately different
question from concentration, and the two are confirmed DECOUPLED.** Added
`device.layer_hits`/`layer_misses` (moe-cache.cu, indexed by `node->layer`,
already parsed from the tensor name at `moe_cache_begin()`), reported via
`GGML_CUDA_MOE_CACHE_SUMMARY`. Result: miss rate is flat across all 40
layers (12.1%-16.0%, no outliers) - and critically, layer 0 (the
concentration outlier) has a completely ordinary miss rate (12.55%, mid-
pack). This is the mechanism explanation, not just correlation: TODAY,
every expert access computes the FULL neuron width regardless of that
expert's own concentration, because nothing yet exploits neuron_heat to
skip work - so of course current cost is flat, the system doesn't yet act
on the signal that says it doesn't need to be. Confirms concentration
(the structural property) and throughput cost (the current, unexploited
behavior) are different questions - "which layer/expert to prefer" needs
the concentration signal for the NEURON-subsetting mechanism specifically,
not a proxy for present-day cost.

**6. Gemma4 (this fork's MTP draft model) has genuine per-layer
heterogeneity Ornith doesn't** - `has_expert = layer.ffn_gate_inp !=
nullptr` is evaluated per-layer (`src/models/gemma4.cpp:128`), some layers
dense-only, some MoE. Not tested (no immediate need once the per-expert,
layer-agnostic design was confirmed correct), but noted as a real
candidate if per-layer heterogeneity ever needs testing again without
GLM-5.3.

### What's built and verified (safe, live, in production on this branch)

- `device.neuron_heat` (moe-cache.cu ~line 598): per-neuron accumulated
  `|value|`, keyed by `(tensor, expert)`, accumulated in
  `moe_cache_collect()` from data it already downloads (`h_out`) - no new
  sync, no new kernel. Gated by `GGML_CUDA_MOE_CACHE_NEURON_HEAT`, off by
  default.
- `device.neuron_heat_mu`: dedicated mutex, NOT `session.mu` - the first
  cut reused `session.mu` and measured a real regression (~50-60 tok/s ->
  ~4 tok/s, GPU util 0%) from contending with the cache's core bookkeeping
  lock. Fixed by giving it its own lock.
- `moe_cache_neuron_heat_report()`: full concentration/required-width/
  dead-fraction report with per-neuron dump capability
  (`GGML_CUDA_MOE_CACHE_NEURON_HEAT_DUMP`), called ONLY from
  `moe_cache_session_destroy()` - deliberately off the hot path, after a
  second real regression was found and fixed (the report's per-key sort,
  first written to run inline from the periodic stats logger, cost
  tok/s down to ~2 when `GGML_CUDA_MOE_CACHE_STATS` was set - moving it
  out of the lock wasn't enough, because the problem was the analysis
  running synchronously on the compute thread at all, not lock
  contention).
- `moe_cache_select_reduced_indices()`: pure function, given one expert's
  heat vector and K, returns the top-K neuron indices by value (ties
  broken arbitrarily), sorted ascending, gated on a minimum accumulated
  mass (`GGML_CUDA_MOE_CACHE_NEURON_REDUCE_MIN_MASS`, default 50) so a
  just-admitted expert with 1-2 samples can't get reduced on
  near-nothing.
- `device.reduced_indices`: storage for the per-expert index set once
  selected, protected by `session.mu` (consistent with pool/slot state),
  lock order `session.mu` -> `neuron_heat_mu` when both are needed (same
  order `moe_cache_neuron_heat_report` already uses from
  `moe_cache_session_destroy`).
- Live UI integration: `moe_cache_get_neuron_concentration_map()` (a
  cheap O(n) dead-fraction proxy, no sort, safe to poll every ~1.5s),
  wired through `common_moe_cache_get_neuron_concentration_map()` ->
  `/experts`'s `neuron_concentration` field -> Brain.svelte, mapped to
  ATLAS DOT SIZE (not color - color already carries tier + substitute
  state). Verified: `npm run check` 0 errors, correctness re-confirmed
  bit-identical output with the flag on/off after every change.
- Layer-level throughput signal: `device.layer_hits`/`layer_misses`,
  reported via `GGML_CUDA_MOE_CACHE_SUMMARY` - real, live, no gating
  issues found.

None of the above changes any existing behavior. Everything is
observation-only or read-only until the mechanism below is built.

### What's NOT built - the precise, ready-to-execute spec for the reduced-width mechanism itself

**Scope, corrected per the finding above: per-expert, every layer, NOT
layer-restricted.** Gate/up only (down_proj deferred - its neuron axis is
the Q4_K quantization-block axis, not row-addressable without a one-time
repack to a neuron-major layout).

**Why this can't be built incrementally the way everything above was:**
the moment `moe_cache_plan()` starts producing `slot_indices` values that
mean "this hit is in the reduced pool", `.dispatch()`/`.collect()` MUST
already know how to decode them, or those values get used as raw
primary-pool slot numbers - out-of-bounds slot access / garbage weight
data / silent corruption or a crash. There is no safe partial state
between "plan encodes, dispatch doesn't decode yet" - unlike
`neuron_heat`, which was safe to build in small additive pieces because
nothing downstream read it. This has to land as ONE atomic change,
behind a flag that defaults off, encoding + decoding + scatter all
together, correctness-verified (coherent generation on real prompts, not
just "compiles") before it's trusted at all.

**The design, precisely:**

1. **Reduced pool creation.** No new pool TYPE needed - `moe_cache_find_pool`
   already keys pools by `(expert_size, wtype)`; a pool with a smaller
   `expert_size` (K rows instead of the full `moe_intermediate_size`) is
   automatically a separate pool via the existing mechanism. K is a fixed
   neuron COUNT (not a percentage - percentages don't mean the same thing
   across experts with different required-widths, and K must be uniform
   across every expert sharing that reduced pool for `ggml_mul_mat_id`'s
   batching to stay legal - confirmed structural requirement,
   `ggml/src/ggml.c:3329-3353`). `GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K`
   already exists as a knob (default 256), reused here.

2. **Conversion trigger.** An expert is ALWAYS admitted normally (full
   width) first - this is also what lets `neuron_heat` accumulate real
   data for it (heat is only recorded on GPU-side hits, via
   `moe_cache_collect()`). Once `moe_cache_select_reduced_indices()`
   returns a non-empty set for a resident, full-width expert (enough
   accumulated mass), it becomes eligible for conversion: gather its
   top-K rows from the SAME host source the normal fill already reads
   from (`host_base + expert * full_expert_size`) into a staging buffer,
   copy to a newly-allocated reduced-pool slot (reusing the existing H2D
   copy pattern, just a different, gathered source), record the mapping
   in `device.reduced_indices[key]`, then free the original full-width
   slot back to its pool's free list. NOT YET DECIDED: where exactly this
   trigger should live - candidates are a new job type in
   `moe_cache_worker`'s existing queue (touches the most fragile,
   corruption-history-bearing part of this codebase, needs careful study
   of its existing job/lock/stream invariants first) versus some other
   safely-isolated trigger point. This is real, undecided design work,
   not just an implementation detail.

3. **Lookup encoding, in `moe_cache_plan()` (~moe-cache.cu:7916,
   `pool.map.find(key)`).** On a miss against the primary pool, also
   check `device.reduced_indices` (and the reduced pool's own map) for
   this key. If found there, encode `slot_indices[index]` as a value that
   cannot collide with a real primary-pool slot number (e.g. a fixed
   large offset + the reduced-pool slot number - primary slot numbers are
   bounded by `pool.n_slots`, so any offset larger than the largest
   possible pool size is unambiguous). `ggml-cpu.c` never needs to know
   this encoding exists - it already treats `slot_indices` as opaque data
   moe-cache produces and consumes.

4. **Batch splitting and scatter, inside `.dispatch()`/`.collect()` ONLY
   - `ggml-cpu.c` stays completely unmodified.** `.dispatch()` splits the
   `n_hits` batch by the encoding (primary vs. reduced), runs the matmul
   against each pool's own slab data (two kernel-launch sequences instead
   of one, for a batch containing both), and for the reduced-pool subset,
   writes each row's K computed values into the correct positions of a
   ZERO-INITIALIZED n_out-wide row (using that specific expert's stored
   index mapping from `device.reduced_indices`) rather than a contiguous
   K-wide write. Once this scatter is done, `d_out` holds correctly-shaped,
   correctly-valued n_out-wide rows for both primary and reduced hits
   alike - `.collect()` needs NO changes at all, it already just copies
   `h_out` back to the caller assuming full n_out width, which is now
   true either way.

5. **`down_proj` is untouched, deliberately, in this phase.** The
   zero-filled positions in the scattered row are exactly what makes this
   correct without touching `down_proj`'s own matmul at all - down_proj
   sums over a mostly-real, partly-zero input vector, which is
   mathematically exactly "the neurons we didn't compute contributed
   nothing", the intended approximation.

**Honest open questions, not yet resolved:**
- Exact trigger mechanism for conversion (point 2 above) - the biggest
  undecided piece.
- Whether eviction/re-expansion needs to be symmetric (can a reduced
  expert ever convert BACK to full width if its concentration profile
  turns out to have been wrong, e.g. from too little data at conversion
  time)? Not designed.
- Quality cost has never been measured for this mechanism at all - no
  perplexity check, no output-coherence check beyond "the observation
  infrastructure doesn't corrupt anything", because the mechanism that
  would actually change output has never been dispatched.
- No tok/s claim anywhere in this section, deliberately - see the D2D
  prefill double-buffer precedent (this doc, "moe-cache prefill double-
  buffer") where a data-movement reduction measured 5% SLOWER due to
  added bookkeeping overhead. This mechanism's overhead (batch-splitting,
  scatter) is real and unmeasured; "fewer bytes moved" is not the same
  claim as "faster", and the second one needs its own measurement once
  built.

## Heat-aware neuron subsetting SHIPPED, and the CPU bottleneck root-caused (2026-08-29, later session)

Continuation of the two sections above. The mechanism specified there is now
built, measured, and working; separately, chasing its disappointing initial
throughput led to a much larger, unrelated finding about placement.

### The mechanism: built and working

All the pieces the previous section listed as "NOT built" now exist in
`ggml/src/ggml-cuda/moe-cache.cu`: `moe_cache_try_reduce_convert` (the
conversion trigger, called from `moe_cache_collect`), a `reduce_convert`
job kind gathering top-K rows in `moe_cache_worker`, the
`MOE_CACHE_REDUCED_SLOT_OFFSET` encoding in `moe_cache_plan`, batch
splitting in `moe_cache_dispatch`, and `moe_cache_reduce_scatter_kernel`
writing K dense results into their sparse positions in an otherwise-zeroed
n_out row. `ggml-cpu.c` is unchanged, as designed. Pins now carry their own
pool pointer (`moe_cache_pin::pool`) since a pin can point into either the
primary or the reduced pool.

Verified: multi-topic, multi-turn conversation (topic switches, context
recall, arithmetic) with correct output, no guard breaches, no corruption
markers. Measured 50% VRAM saving on every converted expert (K=256 against
Ornith's 576 KiB gate/up experts).

### Three bugs, all found by measuring rather than by reading the code

Worth recording as a pattern: every one of these looked fine on inspection.

1. **`session.mu` on the hot path: -16% tok/s** (35.56 -> 29.89).
   `moe_cache_try_reduce_convert` took the cache's core lock on every
   gate/up hit, even though the answer was almost always "already decided".
   This is the same regression shape `neuron_heat_mu` was created to avoid,
   reintroduced. Fixed with `device.reduce_decided`, a keyset guarded by
   the already-held `neuron_heat_mu`, so the common case never touches
   `session.mu`.

2. **20,201 conversions decided, 0 placed.** Pure overhead, zero benefit,
   completely silent. Primary admission (`moe_cache_build_pending`) divides
   ALL remaining budget among pending shapes, so the reduced pool's own
   `moe_cache_allocate_pool` call never saw headroom. Fixed by reserving
   off the top - and then that fix had its own two-stage failure, below.

3. **Reduction was being applied to `down_exps`.** The design says gate/up
   only (down_exps's row axis is hidden_size, an OUTPUT dimension - zeroing
   "unselected" rows there deletes real output dimensions rather than
   approximating activation sparsity). The restriction was never actually
   coded. Fixed with `moe_cache_node::reduce_eligible`, an allowlist on the
   tensor name computed once in `moe_cache_begin`.

The reservation fix in (2) then failed twice more, each time caught only by
looking at the pool table in `GGML_CUDA_MOE_CACHE_SUMMARY`:

- Flat 256 MiB out of a 512 MiB budget starved Ornith's 840 KiB down_exps
  pool out of existence entirely. Hit rate 31.3% -> 23.1%, costing far more
  than the 72 MiB saved.
- Subtracting a per-shape floor did NOT fix it: `moe_cache_build_pending`
  runs repeatedly as shapes are discovered, so on the call where only one
  shape remains pending, the floor covers only that shape. down_exps stayed
  absent, hit rate fell further (19.5%), ~236 MiB of budget went unused.
- Fixed by ALSO capping the reservation at budget/8, which is invariant to
  call count and discovery order. Result: all three pools present (576 KiB
  / 840 KiB / 288 KiB reduced), 511 of 512 MiB used.

With that fixed, at matched `-ncmoe` and matched warm-up: 44.07 -> 47.62
tok/s, **+8.1%**.

### The much bigger finding: the fit margin was the CPU bottleneck

Chasing why the mechanism only bought 8%, the utilization data said
something was wrong with the whole picture: **CPU ~5.4 of 12 cores busy,
GPU only 41% utilized, and ~3 GB of a 12 GB card unused.** Neither resource
saturated means the bottleneck is serialization, not capacity.

Root cause, in `common_maybe_raise_moe_for_ctx` (`common/common.cpp`):

```
const int64_t margin = 3 * (int64_t) params.fit_params_target[0];
```

`fit_params_target` defaults to 1024 MiB, so placement reserves **3 GiB**
before deciding anything - and that reservation silently RAISES a requested
`-ncmoe` until the context "fits". This invalidated an entire `-ncmoe` sweep
before it was noticed: requesting 22, 16 and 10 all produced identical
8947 MiB and near-identical throughput, because all three were being raised
to 27. The log line saying so was there the whole time
(`... raised to 27 layer(s) so the requested context fits`).

Measured on Ornith-1.5-35B-Q4_K_M / RTX 3060 12 GB / -c 4096, requesting
`-ncmoe 8`:

| -fitt | effective ncmoe | tok/s | CPU% | GPU mem |
|---|---|---|---|---|
| 1024 (default) | 27 | 47.32 | 472 | 8951 MiB |
| 640 | 24 | 50.37 | 438 | 10307 MiB |
| 448 | 23 | 51.61 | 445 | 10741 MiB |
| 320 | 22 | 52.85 | 406 | 11237 MiB |

**+11.7% from the margin alone**, ~+30% against the session's starting
`-ncmoe 30` configuration. Every candidate passed its correctness check.
The margin is not removable (it covers post-probe allocation - real weight
loading, lazy CUDA graph capture - and removing it reproduced a documented
4k context collapse), but 3 GiB is not the right value for every machine.

### Calibration upgraded - and it is genuinely model-specific

`common_moe_calibration_entry` now carries `fit_target_mb`,
`neuron_reduce_k` and `neuron_reduce_budget_mb`; the margin is applied in
`common_maybe_raise_moe_for_ctx` BEFORE placement is computed (applying it
after would record a margin that never influenced anything). Old entries
still load - missing fields default to -1. Verified end to end: a launch
with no tuning flags logs both "using calibrated fit margin of 320 MiB" and
"using calibrated placement of 22 CPU layer(s)", and delivers 52.60 tok/s.

**The tightened margin is NOT universally good, which is the whole argument
for calibrating it.** Real `--moe-calibrate` run on gemma-4-26B-A4B,
concurrency 4: the cache knee found 71.86 tok/s at 4096 MiB with the
DEFAULT margin, while every tightened margin scored ~50 (640: 50.79, 448:
51.21, 320: 49.86, 256: 49.56). Opposite of Ornith. A constant would have
been wrong for one model or the other no matter which constant was chosen.

That run also exposed a bug in the new search itself: it seeded its
comparison baseline from `best_threads_tps` (measured at a different cache
size) rather than the cache-knee winner, so it declared fitt=448 the winner
against 43.57 and never compared against the 71.86 the default margin
actually achieved. It would have shipped a ~29% regression as a "calibrated
optimum". Fixed by seeding from `best_cache_tps`, so -1 ("keep the
default") correctly survives when tightening does not help.

Also measured, incidentally: `n_threads=12` lost to `n_threads=6` on this
12-core machine at both concurrency 4 (43.57 vs ~48) and concurrency 8
(75.22 vs 98.59), independently confirming the `n_threads: 6` that older
cached entries already carried.

### Still fixed constants that should be calibration-derived

Named here because they are the same class of problem the margin turned out
to be, and at least one of them is now known to cost real throughput:

- `MOE_CACHE_MIN_POOL_SLOTS` (64) - was a bare literal in three places,
  now one named constant, still a guess.
- The reduce reservation fraction (budget/8) and
  `GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K` (256).
- `cache_candidates_mb[]` in the cache-knee search - a hardcoded ladder.

### Not wired, and the most interesting remaining idea

A converted expert costs K rows instead of full width, so the same hit rate
now needs LESS cache VRAM than it used to. That surplus should feed back
into placement (fewer CPU layers, less CPU work). Nothing does this: the
fit logic still sizes everything as though every expert costs full width.
Reduction currently makes the cache cheaper without ever telling the
placement decision that it did.

## Co-activation persistence shipped, and the discovered atlas's layer-chain bug found and fixed (2026-08-29, later session)

Closes item 11 of the pending queue above (persist co-activation across
restarts). Separately, chasing a visual complaint about the discovered
atlas's layout ("arranged like spokes on a wheel, not fluid") led to a real
structural bug in how the co-activation graph is built, not just a display
issue.

### Co-activation persistence - built

`moe_cache_coact_save` (`moe-cache.cu`) previously overwrote
`GGML_CUDA_MOE_CACHE_COACT_FILE` from `device.co_activation` fresh every
save, unmerged - since `device.co_activation` starts empty every process
launch, each run's file only ever described that session, never
accumulated. Fixed: a new `session.coact_history` map, loaded once (format
bumped to `moe-cache-coact 2`) and added to on every save, keyed by packed
`(layer_a, expert_a, layer_b, expert_b)` rather than by `moe_cache_edge`
(tensor pointers aren't stable across restarts, same reason the predictor
persists by tensor name). Both `device.co_activation` (within-layer) and
`device.co_activation_cross_layer` are written now, not just the former -
the cross-layer signal is required for the consumer
(`scripts/moe-atlas-evolve.py`) to keep the graph connected at all (see
below).

### The atlas layout bug: cross-layer edges were a rigid chain, not real topic structure

Diagnosis: `corr(layer, x) = 0.63` in the discovered atlas's 2D output,
and per-layer angle spread of σ≈0.01-0.05 rad while the mean angle marched
steadily layer to layer - every expert in a layer sat on almost the same
ray, one spoke per layer. Root cause in `scripts/moe-atlas-evolve.py`'s
`fold_in`: cross-layer edges only ever linked layer L's top-1 pick to layer
L+1's top-1 pick, one edge per token step, always to the immediately next
layer. That makes each layer a tightly-connected clique (from within-layer
co-firing) joined to its neighbours by a single narrow bottleneck edge - a
chain of clusters. Spectral embeddings of that exact shape produce leading
eigenvectors that are low-frequency waves along the chain ("which layer am
I in"), regardless of edge weight - confirmed weight was NOT the cause:
within-layer edges carried 28x more total weight than cross-layer (25.4M
vs 0.9M) and were still dominated by the layer-chain topology, because the
effect is structural (which edges bridge clusters), not magnitude-based.

Fix: cross-layer edges now link a token's top-1 pick at EVERY layer it
touched to its top-1 pick at every OTHER layer that same token touched
(all pairs within a token's routing path, not just adjacent-layer), via
token-id grouping (`load_traces`/`fold_in`, batches detected by layer
number decreasing). Regenerating from the 12 real trace files
(`traces/big-*.txt`, 909,480 decisions - `traces/big-progress.txt` turned
out to be a line-count manifest, not trace data, and had to be excluded)
produced a materially different, healthier eigenvalue spectrum: top 12
magnitudes went from near-degenerate (0.9996, 0.9996, 0.9983, ...,
symptomatic of the chain fragmentation) to a real dropoff (1.0, 0.6941,
0.6825, 0.666, ...). Post-fix `corr(layer, x)` = -0.08, `corr(layer, y)` =
-0.04 - essentially gone.

Also added: explicit per-layer mean-centering of the raw embedding before
the disc mapping, stripping whatever layer-correlated residual is left
(the Grid panel already shows layer directly, so the atlas encoding it
again is redundant even after the topology fix).

### Positioning rule changed to match the old fixed-9-topic atlas's "pull," generalized

Requested directly: the discovered atlas should position experts by the
same weighted-combination-of-topic-anchors rule the old fixed-9-category
atlas used (`ec1354bc1`'s "an expert's position is the convex combination
of its category anchors weighted by cats[]"), but with neither the topic
count fixed at 9 nor the anchor positions preset - and by design the
discovered axis count is meant to vary run to run as the graph accumulates
more data (the eigengap heuristic re-decides k from the eigenvalue
spectrum every run, per the module's own docstring), not stay fixed the
way the old atlas's 9 preset categories did.

Implemented as reciprocal averaging: topic anchors for however many axes k
the eigengap heuristic picked this run start on an equal-angle circle
(same mechanism the old fixed atlas used, generalized to k instead of 9),
then get refined over two passes - each topic's anchor moves to the
weighted centroid of the experts that load on it (weight = squared
loading, direction doesn't matter for a spatial pull), and each expert's
position is the weighted combination of the (now-refined) topic anchors.
Topics whose experts overlap drift toward each other; topics with nothing
in common drift apart - unlike the old preset categories, which sat at
their initial angle forever regardless of what the data said. This is a
static, offline computation baked into the regenerated atlas JSON on each
`moe-atlas-evolve.py` run, not a client-side animation - an earlier
attempt at literal per-frame client-side movement (nudging currently-
firing points toward a live pulse-weighted centroid, eased frame to frame)
was tried and explicitly rejected as the wrong shape ("should not be
rendering and showing movement like animation").

### Brain view polish, same session

- Hit-flash (`Brain.svelte`) previously blended toward pure white
  regardless of tier, discarding the hot/warm distinction during any
  active traffic; now blends toward a brightened version of the cell's
  OWN tier hue instead, so green stays green and blue stays blue, just
  lit up.
- Cell brightness now reflects live neuron activity
  (`neuron_concentration`, 1 - dead fraction) instead of raw selection
  heat, falling back to heat where no neuron data has been collected yet
  for that cell - requested directly, on the reasoning that two equally-
  often-selected experts can differ a lot in how much of themselves they
  actually use per firing.
- Pulse decay retimed from ~1.2s to ~1.6s full fade (`PULSE_DECAY`
  0.94 -> 0.953) - just past `POLL_INTERVAL_MS` (1.5s), so a cell that
  keeps firing every poll never quite goes dark before the next flash
  reignites it, while a one-off flash still fully fades before an
  unrelated later poll.
- `docs/index.html`'s "run it the way it was measured" command now points
  `--expert-atlas-file` at the discovered/evolving atlas
  (`expert-atlas-evolve.json`) instead of the static 9-probe file - the
  Brain view's `atlas.discovered` rendering path existed since
  `8e177c436` but nothing in the documented startup command ever fed it,
  so every session that followed the docs got the static fallback layout
  regardless of what the UI code actually supported.

### Still open

- Whether the reciprocal-averaging topic-anchor layout reads as
  meaningfully more "fluid"/topic-clustered in practice, beyond fixing
  the spokes artifact and the layer correlation number, is a subjective
  call - not measured beyond the diagnostic stats above.
- The full-pairwise-per-token cross-layer edge construction is more
  expensive to fold than the old single-chain-edge version (bounded by
  distinct layer pairs per token, not measured precisely) - fine at the
  current trace volume (909K decisions, well under a minute), not
  stress-tested at larger trace sets.
- A live per-frame movement mechanic was built, tried, and reverted this
  session (see above) - if genuine live responsiveness to firing
  patterns is wanted later, the static topic-anchor approach here is not
  that; it only changes what the static layout looks like.
