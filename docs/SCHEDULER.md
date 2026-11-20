# One expert scheduler: what it replaces, what fought what, and how the pause experiment fits

HotRod 21 Nov 2026. Written after the audit asked for on 20 Nov. Every claim says where it comes from: **measured** (a run on the
12 GB RTX 3060 / 31 GB RAM / NVMe 3.2-3.5 GB/s box, Qwen3.8-Flash-Next UD-Q4_K_XL, with the date), **code** (read in this tree), or
**simulated** (tools/sysim, a model, not a measurement).

## The problem in one paragraph

Where an expert's bytes live and when they move is decided in about ten places that do not know about each other. A decode step waits
about 54% of its time on ~45 expert fetches (**measured** 14 Nov: 22-32 missing blobs a window over ~14 of 48 layers, read wait 45-60 ms
of a 105-120 ms window), and 54-60% of the guess reads are never used (**measured** 14 Nov: ~85% wrong guesses, ~14% late). Several
single-purpose switches were tried to tune this one at a time, and all lost or tied (same note). That is the pattern of a system whose
parts fight each other: each switch moves one part, and the others take back what it gained.

## The decision sites today (code)

| # | Site | What it decides | Where |
|---|---|---|---|
| 1 | Residency swaps (`adapt`) | which resident expert leaves the GPU cache for which missing one, by use counts; up to `--adapt-swaps` a window; the copy runs a window ahead | generate.cpp `adapt`, `resident_stage_swaps`, `apply_pending` |
| 2 | PCIe share | which of a layer's misses are copied to VRAM and computed there (`STRATA_PCIE_BALANCE`: from measured CPU and GPU costs) | expert_source.cpp `expert_pool_dispatch_multi`, `pcie_model.pick` |
| 3 | CPU pool | the rest of the misses, computed on the CPU from RAM | pool.cpp |
| 4 | Stand-ins | a missed expert at rank 3-9 is replaced by a resident one (`STRATA_ROUTE_RESIDENT`, the budget it is allowed to cost, learned) | native_router.cu `route_resident`, the budget tuner |
| 5 | Tail skip | a missed expert that every token routes at rank 7 or lower is dropped (output changes) | `tail_skip_rank()`, same dispatch |
| 6 | Look-ahead | guesses of the next layers' experts (GPU `route_pred`, a learned correction), then `io_enqueue` / `ra_issue` read them ahead into RAM or the page cache | RouterLookahead, expert_source.cpp |
| 7 | File-tier I/O | the read threads, their order (EDF), which guess is "held back (drive busy)" | expert_source.cpp |
| 8 | RAM tier and page cache | which experts are in RAM (`--resident-budget-gib auto`), the exchange when a swap happens, what gives way to the pinned K/V | expert_source.cpp, generate.cpp `give_back_ram` |
| 9 | Prompt loan | how many GPU slots the prompt path borrows and refills afterwards | prefill.cpp, generate.cpp |
| 10 | K/V streaming | which K/V cells stay in VRAM and which in pinned RAM (262K) | kv_stream |
| 11 | Learners | the look-ahead predictor, the stand-in budget, the residency profile (profile-learned.bin), the prompt chunk | learned files in data/ |

## What fought what

Measured (14 Nov unless noted; same box, alternated runs; baselines drift 16.7-20.3 tok/s over 17 runs, so a difference under ~10% is
inconclusive):

1. **Guess reads against demand reads, for one drive.** A guess read in flight adds ~0.65 ms to each demand blob (1.28 ms per blob with
   no read-ahead against 1.94 with guesses) - but with no read-ahead there are 62 misses a window against 25. Pausing the guesses while
   a demand read runs (`STRATA_IO_DEMAND_PAUSE=1`) cut the wait per blob 1.95 -> 1.50 ms and the guesses that landed in time fell
   31% -> 10%: decode 18.4-19.0 against 16.7-19.9, a tie. Pausing is all-or-nothing, so what it saves on one side it loses on the other.
   `=auto` tracked the baseline (18.4 vs 18.55). Pause + `PF_THREADS=auto` 16.5, pause + 4 threads 17.2, baseline 18.35: worse.
2. **Stand-ins against the read-ahead.** The look-ahead skipped 582,256 guesses "a stand-in will serve them" against 2,478 queued
   (20 Nov log, **measured**, the guesses at ranks a stand-in covers are never read), and 17,651 were held back for a busy drive. So
   stand-ins decide most of what the read-ahead used to; the two share no estimate of the same probability.
3. **Stand-ins and swaps against the miss path.** Turning stand-ins off: decode 6.0 / 8.2 / 9.8 / 5.4 against 16.1 / 19.7 / 22.9 /
   15.8 (hit rate 93% -> 68%). Turning the swaps off: 13.3 / 11.1 / 11.0 / 11.7 (21 Nov). Both matter a great deal on this card, and
   neither knows what the other is covering.
4. **Four switches that did nothing on this card** (21 Nov, within the ~1 tok/s noise of two base runs): the route tail skip, the learned
   look-ahead, the mask, the CPU pool's one-batch layer call (`STRATA_POOL_DEP`). They cost nothing that shows on a card that misses many
   experts; on a card that misses few they may cost a fixed time per window (the 24 GB review: decode 103 -> 87 tok/s, prefill up).
   That is why the calibration now measures each against its off state (tools/calibrate.py).
5. **A mid-window admission against the tier's exchange.** Upstream's `STRATA_ADAPT_FETCH` copies a PCIe-fetched expert straight into
   a victim's slot, so the expert is resident next window (upstream measured +10.2% at a 16% cache). Here it cannot be a local change:
   in the resident RAM mode an expert evicted from the GPU cache must be written back into the RAM entry of the expert that replaces
   it (`commit_exchanges`), and the device residency table is patched only by the tier's own hook. An admission inside the window would
   need both done mid-window, by code that today belongs to the swap path. (**code**; the port was stopped here on 21 Nov, not tried.)

From the earlier notes (**code** / **simulated**): a mask, a prompt loan that takes slots the swaps then refill, a RAM tier and a pinned
K/V that compete for the same pages, and a PCIe lane that the swaps and the PCIe share both use.

## The design: one owner, one currency

**One place owns the invariant** "this expert is in a GPU slot, a RAM entry, or only in the file, and nothing else moves it". Swaps,
admissions, the prompt loan and the RAM exchange become requests to it. That removes the arguments in 5 and the hidden ordering between
`adapt`, `apply_pending` and the pre-launch hook.

**One currency for every move: expected stall saved per byte and per millisecond of the link it uses.** Each candidate move - a guess
read, a swap-in, an admission, a prompt refill - is scored by

    P(used before it is evicted) x (stall it removes: the demand read, or the CPU compute, it replaces) / (bytes x link cost)

- `P(used)` comes from the look-ahead's own calibrated probability (the learner already trains a rank/true-rank loss; its output is
  made a probability by a held-out calibration, the way the draft head's acceptance chance is), and from the stand-in rule: a guess a
  stand-in will serve has benefit about zero, which is today's "skipped (a stand-in will serve them)" made one number instead of a flag.
- The deadline is the layer's position in the window: a guess for layer l+4 can wait, one for l+1 cannot.
- **Demand reads are not paused; they are scheduled first** - the drive is a measured shared resource (~3.2 GB/s for batches of 1-3
  blobs, ~1.0 ms per 3 MB blob), the demand read has a hard deadline and the highest benefit, and a guess starts only if the drive's
  model says it finishes before the demand work that follows it and its score clears the threshold. This is what the pause tried to buy
  with a switch. The measured cost of the pause (guesses landed 31% -> 10%) came from refusing every guess during a demand read;
  a deadline test refuses only the guesses that would be late.
- A cost-aware threshold is the idea the simulator already priced (**simulated**, tools/sysim, 22 Oct: reading a guess only at
  confidence >= 0.6 is +18% short / +29% long on today's setup and +10% / +15% on the fixes). It is a model, to be measured before it is believed.

**What it replaces:** `STRATA_IO_GATE`, `STRATA_IO_PF_YIELD`, `STRATA_IO_DEMAND_PAUSE`, `STRATA_IO_PF_THREADS` as numbers someone sets
(the thread count and order come from the drive model), the "held back (drive busy)" counter's ad-hoc test, the separate swap budget
`--adapt-swaps` (a swap is one more scored move, limited by the PCIe link), and the stand-in/read-ahead split in (2). **What it does not
replace:** the stand-ins themselves (they are compute alternatives to a read, and the scheduler asks them for their cost), the CPU pool,
the learners (they feed it), the K/V streaming rule.

## Build order, each step gated by a measurement

1. **Instrument before building.** The decode timeline already records plan, fetch, CPU run and GPU gap per layer. Add, per
   guess read, `P` at issue time and whether it was used, so `P` can be checked for calibration (is a 0.6 guess used 60% of the time?).
   Gate: a reliability curve on the 3060, from the real run.
2. **The drive model.** Replace `PF_THREADS` / gate numbers by a queue that knows ~1.0 ms per 3 MB blob and ~3.2 GB/s for 1-3 blob
   batches and starts a guess only if its deadline holds. Gate: wait per demand blob at or under the no-read-ahead 1.28 ms while the
   landed-in-time share stays at the 31% of the baseline or better.
3. **Placement through one owner.** Move `adapt` / `apply_pending` / the exchange behind one interface with the invariant above; no
   behaviour change. Gate: bit-identical greedy output, the same swap counts.
4. **Then, and only then, the admission** (the upstream +10% idea), as one more move through the owner. Gate: a decode gain on this
   card larger than the ~10% noise over two alternated pairs, and the quality read by hand.
5. **Calibration measures the scheduler's few constants** (the score threshold) on each PC, as it does for the other switches.

## What was not done and why

- The pause experiment was not re-run: the 14 Nov note says do not, and the data above already shows its trade-off.
- Nothing here changes a default. The scheduler is a plan; the measured parts are the evidence for it.
