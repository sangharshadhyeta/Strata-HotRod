# Do the features support each other, or work in silos?

Audit of 2026-09-11. The question came from a simple observation: each subsystem
here is individually measured and individually defensible, yet several of them
had spent the day quietly fighting one another.

The answer is not uniform. Two cross-component pipelines are genuinely well
connected. Two subsystems are completely disconnected. And the most expensive
failures of the day were not missing connections at all — they were components
that *were* connected, passing each other numbers that did not mean the same
thing.

---

## 1. Inside the moe-cache: thirteen signals, each decision reads a few

Every decision below runs on the same device state. The columns are what each
one actually reads.

| Decision | Reads | Notably ignores |
|---|---|---|
| `plan` (per-token routing) | VRAM heat, RAM heat, atlas, req_dir, predictor, co-activation, neuron heat, pins | cost tier |
| `atlas_admit` | atlas position, `is_cold`, pins | RAM heat, cost tier |
| `cold_sweep` (MADV_COLD) | RAM heat, `last_seen`, `is_cold`, pins | atlas, predictor, cost tier |
| `substitute_pick_hot` | VRAM heat, co-activation | whether the *missed* expert is RAM-cheap; cost tier |
| `host_promote` (RAM tier) | RAM heat only | everything else |
| `relieve_vram_pressure` | free VRAM, external baseline | the cache value of what it frees |
| victim scoring | heat × cost tier + atlas-align bonus | — (the only consumer of cost tier) |

The pattern: **the disk-cost signal exists and is almost unused.** `cost_tier`
answers "would missing this expert hit the NVMe?", which on a model larger than
RAM is the dominant cost. Exactly one decision reads it — eviction. Admission,
substitution and prefetch all ignore it.

It is also gated on another feature's bookkeeping: an expert only reports the
NVMe tier if the cold sweep flagged it `is_cold` *and* `mincore` confirms the
pages are gone. Every expert the sweep has not visited is assumed RAM-cheap.
So the signal that matters most is both under-read and under-populated.

## 2. Across components

**Connected, and working:**

- **Atlas** — 270 references in `moe-cache.cu`, 85 in the server, 19 in
  `scripts/moe-atlas-evolve.py`. The cache emits co-activation, an offline script
  evolves an expert map, the server reloads it live, and the cache scores
  admissions against it. A real loop.
- **Co-activation** — written by the cache, consumed by both the evolve script
  and the server.

**Disconnected entirely:**

- **Prompt cache ↔ expert cache: zero references.** Restoring a long-idle prefix
  tells the expert cache nothing, so it rediscovers that prefix's experts from
  scratch as decode proceeds.
- **Speculative decoding ↔ expert cache: zero references.** A draft model
  proposes tokens whose expert routing is knowable before the target verifies
  them; nothing uses that.

**Connected, but talking past each other:**

- The **loader** decides "this model is bigger than RAM" (`mmap_prefetch`) and
  tells nobody. The **moe-cache** independently reads `/proc/meminfo` and cgroup
  limits for its own budgets. Two separate estimates of the same fact.
- The **scheduler**'s `WILLNEED` pre-pass and the cache's `CPU_PREFETCH` are two
  mechanisms driven by the same routing decision, unaware of each other.

## 3. Where the real damage was: not missing wires, mismatched units

Every expensive failure found on 2026-09-11 was two components exchanging
numbers that did not mean the same thing.

- **Pinning vs. mmap.** The loader page-locked mmap-backed experts for DMA. On a
  model larger than RAM that broke copy-on-write and turned 18.4 GiB of weights
  into unevictable copies — a correct optimisation for the case it was written
  for, applied to a case that inverts it.
- **Read-around vs. router-driven access.** The kernel's 4 MiB sequential
  read-ahead served 0.88 MiB expert slices in router order: 1.2–1.5 GiB of disk
  per token, most of it never used.
- **Cheap probes vs. confirmed numbers.** Calibration ranked candidates on
  32-token no-reasoning probes but compared them against full-length confirmed
  incumbents. `-ngl`, thread count and the stand-in sigma could not win on merit,
  and a 32-token probe of 15.10 became the cached headline while serving was ~13.
- **A gate measuring the wrong property.** Three quality gates rejected
  *difference* rather than *wrongness* — see the corrections section of
  `index.html`. The cache makes greedy output vary by design; the gates read that
  as corruption and threw out the best configuration nine times.
- **A check that passed something broken.** The stand-in sigma was chosen in the
  reasoning-off regime, where its failure mode cannot exist, then served an empty
  answer and a bare `</think>` with reasoning on.

The lesson is narrower than "connect everything": **a signal is only safe to
share if both ends agree what it measures.** Four of the five failures above came
from sharing a number across a boundary it was not valid across.

## 3b. A rule that came out of getting this wrong twice

Both attempts to consume the disk-cost signal failed the same way, and the cost
model was only half of it.

- **Substitution** is gated on `rank_bucket >= substitute_min_rank`: the router's
  confidence ordering decides what may be stood in for. The gate added residency
  as an extra veto, so an expert the router had already ranked as substitutable
  could be refused a stand-in for a reason the router knows nothing about.
- **Admission** is gated on `demand->count`: an expert earns a slot because the
  router kept asking for it. The bias let residency skip that bar, so a candidate
  could jump the queue ahead of experts that had actually proved they were wanted.

In both the mechanism is an **earned** signal - rank, or repeated demand - and
residency is a **property** a candidate merely has. The rule: a property may
inform how an earned signal is weighed (eviction does exactly this, scaling heat
by cost tier), but it must not gate or bypass the earned signal itself.

That also narrows recommendation 1 below: eviction was not just the only consumer
of cost_tier by accident, it is the consumer where a property *should* act -
weighing, not deciding.

## 4. Calibration reaches nine of the runtime's knobs

Reachable: `GGML_CUDA_MOE_CACHE` / `_MODE` / `_BUDGET_MB`, `SUBSTITUTE_MIN_RANK`,
`SUBSTITUTE_QUALITY_SIGMA`, `NEURON_REDUCE` / `_K` / `_BUDGET_MB`, `MAX_BATCH`.

Not reachable, and therefore never measured on any model: the cold sweep, atlas
warming, the trained predictor, group admission, coverage eviction, the host
hot-expert buffer, CPU prefetch, expert read-around, the scheduler `WILLNEED`
pre-pass, LFRU D2D, and lookahead depth.

Several of those ship off by default with comments saying they were "measured
harmful" — on a different model, under a different memory regime, before this
week's fixes. They cannot be re-measured today without editing source.

## 5. What is worth doing, in order

1. **Populate the disk-cost signal — and know where it is valid.** The
   population half is done: residency is now sampled for every expert (one
   `mincore` per tensor per cold sweep) instead of being inferred only for the
   minority the sweep had flagged. Measured effect on throughput: *cannot
   resolve* — medians 12.83 vs 13.18, IQRs overlapping, rounds split 2/4.
   Populating a signal that only one decision reads does not move anything.

   The consumption half was then tried at substitution and **is wrong**. Gating
   substitution on "is this expert resident?" measured 8.02–9.14 tok/s against
   11.45–12.97 with the gate off — a ~30% regression against a ~5% noise floor.

   The cost model was the error, and the correction is the useful output of this
   item: `cost_tier` answers *"what does it cost to **fetch** this expert into
   VRAM"*. Page-cache residency removes the fetch cost but not the **compute**
   cost — the exact expert still has to be multiplied on the CPU, which is far
   slower than serving a resident stand-in from VRAM. So the signal is valid for
   **fetch** decisions (eviction, which already uses it; admission; prefetch) and
   invalid for any decision whose alternative is *computing* the expert rather
   than fetching it. Substitution ignores this signal **correctly**.

   Remaining and untested: admission, which is a genuine fetch decision and the
   one place the original recommendation may still hold.
2. **Let the loader tell the cache what it already knows.** One
   "bigger-than-RAM" decision, made once, consumed by both — instead of two
   independent estimates that can disagree.
3. **Make the remaining knobs calibratable** before trusting any "measured
   harmful" default that predates this week.
4. **Prompt cache → expert cache prewarm.** Restoring a prefix should say which
   experts it is about to need.

Deliberately *not* recommended: wiring the router lookahead into more models. It
was tried on qwen4exp on 2026-09-11 in all three ways the prediction can be
consumed, and lost to not producing it every time — the pool runs saturated and a
speculative fill only takes free slots. Connecting two components is not
automatically an improvement, which is the same lesson as section 3 from the
other direction.
