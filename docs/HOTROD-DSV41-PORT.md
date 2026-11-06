# Strata-HotRod on DeepSeek-V4.1-Flash

Everything Strata-HotRod built for Qwen3.8-Flash-Next, ported to upstream Strata's DeepSeek engine (`strata-dsv41`, `src/dsv41`, from
Project Maya's maya-ds4). Branch `upstream-dsv41`, release `v0.2.7.1-hotrod.1`. **Run and measured on the model** (31 Oct 2026).

The model: Starw1's DeepSeek-V4.1-Flash-1017 uncensored, the GSQ-RCO IQ2_XXS backbone (138.6 GB; its experts IQ1_M / IQ2_XXS /
IQ2_XS per layer, 1.94 bits a weight) with DeepSeek's own MXFP8 Engram (202.8 GB), the DSpark drafter (8.0 GB, Q8_0, a separate file:
the backbone has no drafter tensors) and Unsloth's 5.8 MB metadata shard. Setup knows it as `--family deepseek41 --model
Starw1-2.00bpw-MXFP8`. The PC: RTX 3060 12 GB, i5-12400F, 31 GB RAM, PCIe 3.0 NVMe.

## Speed on this PC (answer tokens/s, the same 3 questions, 32K context, every answer read)

| Setting | tokens/s |
| --- | ---: |
| Upstream as it ships (the learner off) | 5.05 |
| **The default here: the VRAM ring + stand-ins** (`DS4_RING=32 DS4_RING_PIN=1 STRATA_STANDIN=1`) | **5.2 / 5.3 / 5.4** in three runs, against 4.8-5.0 in the same sessions |
| **The default with the ring on: + sticky, the chunk arena as ring slots, in-flight auto** (see Running it, 2) | **4.66-5.01** (auto in-flight 4.88) in the same session set |
| The learner alone (on once its file exists) | 4.9-5.03 (no measurable gain) |
| The ring alone (plain / pinned copies) | 3.4 / 3.6 |
| Stand-ins alone (with the re-check, the tuner, the exact read-ahead) | 3.9-4.2 |
| The DSpark drafter capped at 0.05 / 0.1 of its experts on the card | 4.96 / 4.97 (drafts kept 80-91%) |
| The drafter uncapped | 4.72 |
| K/V streaming at 32K (`STRATA_KV_HOST=1`) | 4.82 |
| Upstream's `DS4_STORE_ZC=1` | 4.32 |

Why: 270 of the 15,360 routed experts (2.65 GB) fit the card next to 6.2 GB of dense weights, so ~75% of the picks are computed on the
CPU lane (from the RAM store or in place from the files); the PCIe link sits ~99% idle. The ring copies the learner's guesses for the
next 1-2 layers into card slots on its own stream and thread and a pick uses a slot only if its copy has landed; with the stand-ins the
CPU lane's share falls to 34-37%. Each token still needs ~2 GB of expert weights from RAM and the NVMe, which is what the ring cannot
remove (a copy reads the same bytes the CPU would). Upstream's own 3060 figure (9-11 tokens/s) is on 46 GB of RAM.

CPU cost per expert by type (the timeline, 27,195 layer rows): IQ1_M 572 us (74 us/MB, the smallest experts and the slowest), IQ2_XXS
435 us (48 us/MB), IQ2_XS 475 us (46 us/MB). `DS4_RING_COST` uses this live.

## What is ported and how it maps

An expert that is not on the card is computed **on the CPU** here (read from RAM or the SSD in place); the Qwen engine fetches it to
the card. So the ideas carry over, but each is reworked:

| Qwen feature | On DeepSeek | Switch (engine default) |
| --- | --- | --- |
| Stand-ins by confidence | in the router kernel: ranks 0-1 exact, rank 2 within 1 logit, from rank 3 `w x gap <= budget`; the weight kept | `STRATA_STANDIN=1` (off) |
| Exact re-check of a close call | the token run again at the same position without stand-ins | `STRATA_FRAGILE_RECHECK` (on with stand-ins) |
| Self-tuning budget, brake | `ds4_standin_tune.hpp`, `<pack>/standin-tune.txt` | on with stand-ins |
| Learned look-ahead 1 / 2 / 4 layers | `ds4_lalearn.cu`, `<pack>/hotrod-predictor.bin` (learned on your PC; not shipped) | `STRATA_LA_LEARN` (on when the file exists) |
| Read ahead only exact picks | the learned guesses into the RAM store | `STRATA_LA_TAIL_READ=2` |
| **VRAM ring** (new here) | the guesses copied ahead to card slots, used only if landed; `DS4_RING_PIN`, `DS4_RING_RANKS`, `DS4_RING_COST`, `DS4_RING_STICKY` / `DS4_RING_MOVE` (experimental: copies kept across tokens are reused ~6x, but fills starve) | `DS4_RING=<slots>` (off) |
| Drafter's card share (new here) | DSpark's hottest experts on the card, the rest from RAM like the trunk's | `DS4_DSPARK_VRAM=<f>` (all) |
| K/V streaming | the compressed K/V rows in host-mapped RAM | `STRATA_KV_HOST` (on above 131K) |
| Learned VRAM reserve | `<pack>/vram-reserve.txt` | without `--reserve-mib` |
| Request-end reports | stand-ins, re-checks, the budget, the learner, the ring - printed and saved after every answer | on |
| Monitor, stalled-client timeout | ported | on |
| Three tiers, chunked prompts, saved conversations, DSpark | upstream's own | upstream |

Every switch, flag and server key: [FEATURES-deepseek.md](FEATURES-deepseek.md).

## Running it

1. `scl enable gcc-toolset-13 -- ./setup.sh --family deepseek41 --model Starw1-2.00bpw-MXFP8 --gguf-dir <the folder with the files>`
   (setup never deletes your files; it checks them against Starw1's pins). The pack is a few MB beside the files: no weights copied.
2. In the config setup writes, the default set: `"env": { "DS4_RING": "32", "DS4_RING_PIN": "1", "STRATA_STANDIN": "1" }` (a blank
   value means the engine's default). The drafter: a pack made with `--dspark-gguf`, `--dspark on`, and `DS4_DSPARK_VRAM=0.05`.
   With `DS4_RING` set, these are on by default (each turned off with `=0`): `DS4_RING_STICKY` (landed slots kept until needed),
   `DS4_LEND` and `DS4_LEND_RING` (the chunk arena's bytes become ring slots, no fixed experts), and `DS4_RING_INFLIGHT=auto` (the
   in-flight cap climbs on decode speed; a number or `fixed` keeps a fixed cap). `DS4_RING_INFLIGHT=fixed` is the old 12.
   Rule 3 (`STRATA_LA_TAIL_READ=3`, the learned read-ahead) stays opt-in: today's numbers do not show a gain for it.
   Best measured setup on this PC: 4.66-5.01 tok/s; with `auto` in-flight 4.88 (the spread of the runs, same prompts).
3. The learned VRAM reserve counts the ring's slots as planned use (they are taken after the expert placement), so it does not grow
   because of them: 270 experts stay on the card start after start. A reserve raised by an engine before this fix (e.g. 1244 MiB)
   comes down only slowly; set its line in `<pack>/vram-reserve.txt` back once (926 MiB on a 3060 at 32K).

## Tests

| Test | Runs on | State |
| --- | --- | --- |
| `ds4_timeline_test`, `ds4_standin_tune_test` | CPU | pass |
| `ds4_standin_test`, `ds4_lalearn_test` | GPU | pass (31 Oct) |
| `tests/terminal` | CPU | pass (4 need the `textual` package) |

## Rules kept

Answers read by hand (degeneration and degradation reported apart). Never compile during a speed leg. One model on the GPU at a time.
No regex, no pattern matching on tokens. A switch becomes a default only when it wins its A/B and the answers stay clean.

## DS4_LEND (Avalon point 1)

Why: measured 1 Nov, the chunked prompt pass's buffers (`alloc_chunks`) take about 1.76 GB of the RTX 3060. Without them (`--prefill-chunk 0`)
475 experts fit on the card instead of 270, and answers ran 5.9 tok/s instead of 4.7. Those bytes sit idle while no prompt pass runs.

How (opt-in, `DS4_LEND=1`; unset or any other value leaves the engine as it was): the chunk buffers of the trunk's device are carved from
one arena (measured first, one `cudaMalloc`, zeroed). While no chunked pass runs, the arena holds experts: `lend_request` plans them (the
trunk's experts the cards do not hold, hottest first, as many as the arena takes), the lend thread copies each blob in (`parts_to_blob`,
the cache's own layout, on its own stream), and `cache_tick_swaps` publishes the copies into `vram_tab` between windows. Before a chunked
pass (`feed` / `feed_mark`, and again in `forward_chunks` after the warm-up lock) `lend_reclaim` stops the fill, zeroes the table entries
that are still the arena's, and zeroes the arena. After the pass `lend_request` plans again. The swaps and the ring skip lent experts.
Off when: not a CUDA build, more than one device, no chunked path, the expert cache off, or a trunk layer's experts in a group (said once
as `ds4: DS4_LEND: off (...)`). GGUF (MXFP4) expert layers are lent, read from their file as `swap_in_mx` reads them; under copy mode they are not. Logged: `ds4: DS4_LEND: ... experts lent (...) on top of the N in VRAM` at the
first publish, and `ds4: request: lend <lent>/<planned> experts` at each request end.

State: untested until measured. Not yet done: counting the picks served from lent experts.

Measured 1 Nov: DS4_LEND with 117 fixed experts in the 1.02 GB arena gave no gain (5.2 against 4.95 tok/s, inside the spread). The
profile shows DeepSeek's routing is flat, so a fixed hot set adds about 3 points of picks. Its variant, `DS4_LEND_RING=1` (with
`DS4_LEND=1` and `DS4_RING`), takes the arena's bytes as ring slots instead: the ring grows by the arena's whole slots of ring_stride
(about 99 of 10.3 MB on top of the 32), the ring's fills may take them, and a chunked pass takes them back first (no fill is in flight
when the pass starts) and gives them back after. No fixed experts are planned in this mode. The ring's line then ends with ` x <n>`, the
fills that went into arena slots. Untested until measured.

**RAM store under pressure (`ds4_store.hpp`):** below 4 GB free the store gives back its idle chunk with the least use (use: the reads it
served, x0.5 per shrink), not the last one; a given-back chunk is taken back once 6 GB are free (one per 16-window check). Untested here.

**Read-ahead by stand-in failure (`STRATA_LA_TAIL_READ=3`, untested):** mode 2 reads into the RAM store only the guesses ranked below the
stand-in floor. Mode 3 also reads a guess at or above the floor when its stand-in would likely fail (the CPU lane would compute it from
the drive): the kernel records each pick's stand-in outcome (`Req::sub`, `Req::orig`, written only when the stand-ins were evaluated), the
serve thread learns per rank, per card bucket of the layer (`card_n`: the experts with a table entry set now, in buckets 0-7, 8-11,
12-19, 20+) and per expert, and the threshold follows the store's read queue (up 0.02 above half its limit, down below a
quarter). Falls back to mode 2 without the learner, the stand-ins, or the store. Report: `ds4: request: tail-read3: ...`.

**DS4_RING_FAR=1 (opt-in, untested):** with the ring on, `ring_request` also fills from the learner's guesses 4 layers ahead (layer li + 4,
the `Req::ahead` slots after the 1- and 2-ahead ones), after the near guesses of the same request, so the near guesses keep priority. A far
fill takes only a slot of a layer before li (never an ahead slot), so the sticky evict-ahead tiers are unchanged for the near fills. The
slot stays "ahead" until li reaches its layer, and its late/used accounting is per layer as for the near fills. The ring's line ends with
` far <n>`, the fills from the far guesses. Untested until measured.

**Per-distance ring counters (always on, logging only):** each ring slot records the pass that filled it (`RingSlot::dq`: 1, 2 or 4 layers
ahead). When its layer's request comes, the slot is counted per distance: `rd_n` (counted), `rd_pay` (picked and landed in time: the copy
paid off) and `rd_late_pick` (picked but late). The ring's line ends with ` d<1|2|4> <pay>/<n> paid (<late> right but late)` for each distance
with a count. A decayed window per distance (`rw_n`, `rw_pay`, halved together past 512) gives its worth, rw_pay / rw_n.

**DS4_RING_WORTH=1 (opt-in, untested):** implies DS4_RING_FAR. `ring_request` runs its three passes in the order of their worth, highest
first (ties keep 0, 1, 2). Until every pass has 32 counts the order is the fixed 0, 1, 2. Every 8th such request the pass with the fewest
counts goes first, so a last-placed distance keeps being measured. Inside a pass nothing changes: its own q keys the layer offset, the guess
offset, the count and the eviction restriction (`ahead_ok && q < 2`). The request line adds ` order <a>,<b>,<c>` (the distances in the last
order used). Untested until measured.

**DS4_RING_WORTH=2 (opt-in, untested):** implies DS4_RING_FAR. Why: measured 6 Nov, `=1` with DS4_LA_AHEAD2=1 fell from 6.41 to 5.99 tok/s;
its paid shares were d1 24%, d2 35%, d4 54%. Sorting whole passes copies a distance's weak ranks before another distance's strong ones.
So every candidate guess is ranked on its own: each cell (distance, guess rank) has a decayed paid-and-needed worth, cw_val / cw_n (a copy
picked and landed in time is worth its need: 1 for a pick ranked below the stand-in floor or with the stand-ins off, else the stand-in
failure rate `sfail` for that pick's rank, learned from `Req::sub`; an unpicked or late copy is worth 0). `ring_request` collects the guesses of all live passes, sorts them by their cell
(stable; cells under 16 counts score 0.5 and keep the order 0, 1, 2 then rank), and fills each with the same body as before. Every 8th
request the least counted guesses go first. The request line adds ` best d<1|2|4>r<rank> <worth>` for the five best warm cells and
` sfail <r0> ... <r7>`. Unset or `=1`: as before. Untested until measured.

**DS4_STORE_EARLY=N (opt-in, untested):** `ahead_ask` reads into the RAM store, for the guesses 2 and 4 layers ahead only (not the next
layer's), the ranks i < N instead of only the ranks below the stand-in floor (`STRATA_LA_TAIL_READ=2`). Unset or 0: as before. The reads
it starts beyond the mode-2 ranks are counted in ` early <n>` on the ring's request line. These are guess reads, so the auto read-ahead cap
governs them. Untested until measured.

**DS4_STORE_PROTECT=1 (opt-in, untested):** the RAM store does not give up an expert that a guess says is about to be used. Why: measured 6 Nov, `DS4_STORE_EARLY=4` lost 11% and raised the ring's "not in the store" count (80K to 89K). The new guess reads evicted experts the next layers were about to use. `take_slot` gave up the lowest-score slot, and a `prefetch` of an expert already in the store did not refresh it, so an expert guessed for the very next layer could be the oldest and go to a guess four layers out. With the flag, `ahead_ask` gives each guessed expert the position it is for (`store->want`; the token count times the layers plus the layer; `set_pos` once per layer request on the serve thread), and `take_slot` skips a slot whose want is at or past the current position: a guess read is then dropped (`protect drop`), a demand read takes a wanted slot only when nothing else can be evicted (`protect forced`). The ring's request line adds ` protect drop <n> forced <n>`. Unset: as before. Untested until measured.

## DS4_MASK (opt-in, untested)

Why: measured 6 Nov, a decode token is about 70% the GPU waiting for the CPU lane, and about two thirds of the lane's time is waiting for
the RAM store's reads: 8-33 misses a token, about 3.6 ms each (one ~10 MB expert at 2.8 GB/s). Experts the router never picks are never read.

How (opt-in, `DS4_MASK=<fraction>`, e.g. 0.25; unset or 0 leaves the engine as it was): at start, per trunk layer, the fraction of experts
with the fewest picks in the profile are masked (`mask_init`; the profile is the `prof` counts loaded from the pack's profile file, uint32 hits
at `prof[layer * n_exp + expert]`; a layer with no counts is not masked, and with no counts at all the engine says so and runs unmasked). The masks
are one device block on the trunk's GPU (a single-GPU trunk only, as the stand-ins). The router's top-k is taken on the unmasked scores as now,
then a masked pick moves by the stand-ins' ranks (`STRATA_ROUTE_RESIDENT_RANKS`, default 3, and `STRATA_ROUTE_RESIDENT_EDGE`, default 1.0):
ranks below floor - 1 stay; rank floor - 1 moves only when the edge is above 0 and the candidate's logit gap is within it; ranks from the
floor on always move. The candidate is the best unmasked expert the token did not pick. The rule is `route_mask_apply`, used by
`route_publish_kernel` (decode), `route_kernel` (the prompt path's router), and `route_ahead_kernel` (the guesses for a later layer, under that
layer's mask). The learner's labels are the router's raw logits and are not changed (no CUDA call is added on the token path). The
Metal selection and dispatch do not apply the mask. Start-up line: `ds4: DS4_MASK=0.25: per layer the N least-picked experts masked (...),
M of the T masked on the card`. Request end: `ds4: request: mask: p=0.25 F 3 E 1.00, <n> picks moved to the next unmasked expert` (cumulative).

State: untested until measured.

## DS4_STANDIN_LEARN (opt-in, untested)

Why: a stand-in replaces an off-card pick with a resident expert, so its output differs from the true expert's. A correction learned live per
pair lets the stand-ins cover more picks at the same quality, without training the model.

How (opt-in, `DS4_STANDIN_LEARN=1`; unset leaves the engine as it was): the flip re-check (`Impl::step`) already runs a close token again
exactly. Its first pass captures, per trunk layer, the first stand-in's down output y_s (`learn_capture` after `gg::moe_down`, from gguf's
`moe_part`, the per-slot outputs); the record (slot, true expert e, stand-in s) is written by `route_publish_kernel` into `cap_rec`. The
exact re-run (`learn_pass` 2, `standin_bypass`) captures the true expert's y_e at the same slot: from the device when the pick was computed
on the card, from the CPU lane (`serve_lane`, its `c_yd` slot before the weight) when it was not. One sample per re-checked token: the
token's first layer with a stand-in, whose inputs are the same in both passes. The sample updates the pair (layer, e, s) with decay 0.98:
num += y_s . y_e, den += y_s . y_s, c = num / den clipped to [0.5, 2], used after 32 samples. The spec's per-channel c[d] is summed over
the channels: the sum of the per-channel num and den is the scalar pair state, so each pair holds two numbers (no 512 MB cap is needed).
The correction is a scalar on the stand-in's routing weight (`csc`, read from the device table `d_pair_c` in `route_publish_kernel`), since
the combine (`moe_combine`/`mx_combine_kernel`) sums per-slot weighted outputs and a per-channel scale there would be invasive. The table is
[layer][e][s] (1 until ready), at most 4096 pairs a layer (the least recently used goes; its entry returns to 1). Request line:
`ds4: request: stand-in learn: <n> pairs ready, <s> samples this request, mean |c-1| <v>`. Cost on the token path when on: one memset of
the record and one small capture kernel per trunk layer of each decode token (a copy of one slot's D floats on the device), and the
record/sample copies to the host only on re-checked tokens. Needs the flip re-check and GGUF experts on one GPU (the exl3 down fuses the
combine and writes no per-slot outputs, so the learner is off there).

State: untested until measured.

## DS4_STANDIN_BALANCE (opt-in, untested)

Why: a stand-in's cost is its measured error, not the logit gap, and keeping an expert fast costs reads only where it is picked often and
misses. The balance decides both per (layer, e) from measurements, with no hand-set thresholds.

How (opt-in, `DS4_STANDIN_BALANCE=1`, needs `DS4_STANDIN_LEARN`; unset leaves the engine as it was): a pair (layer, e, s) with 32 samples
has err = sqrt((ee - 2 c num + c^2 den) / ee), the corrected stand-in's relative output error from the learner's decayed sums (`ee`, the
true expert's energy, is the one new sum). `route_publish_kernel` exposes a ready pair's stand-in by its err (`d_pair_err`, [layer][e][s],
-1 until ready): its risk is weight x err and, from the floor on, the budget is checked against weight x err; the edge rule at floor - 1
keeps the gap. Every 256 decode tokens (`bal_round`, between tokens) the host ranks each layer's experts by value / cost: value = use(e) x
the best stand-in's err (the router-weighted decode picks, halved each round), cost = the read cost (the store's demand wait per demand
miss, decayed) x the expert's misses (`ExpertStore::miss_counts`). The top decile moves up one state (+1: the store's eviction score is
raised by `set_prio`, and the ring fills its guesses for that expert at any rank); the bottom decile moves down (-1: its score is lowered,
so its slot goes first). Back to 0 only after two rounds on the other side. A stand-in at the soft state needs no separate rule: a
card-absent pick at rank floor - 1 or beyond is already a stand-in candidate under the err rule.

Units: err is a relative output error, the gap a router-score difference, so the budget (tuned by the flip re-check) now compares different
quantities for measured pairs; the tuner adapts it only when the tuner runs. Request line: `ds4: request: balance: <a> pairs measured, <s>
swapped in, <m> soft-masked, read cost <ms> ms/miss`.

State: untested until measured.

## Measured 6 Nov (vision server, 262K, the three calibration questions; answers read by hand, all correct)

The ring opt-ins above, after the auto read-ahead cap became the default. Single runs first, then 3 alternated rounds of every arm
(the default ranged 5.95-6.62 tok/s across rounds, so only the alternated means count):

| Arm | Mean of 3 rounds | vs default |
| --- | ---: | ---: |
| default | 6.33 | - |
| `DS4_RING_WORTH=1` | 6.22 | -1.7% |
| `DS4_RING_WORTH=2` | 6.38 | +0.9% (noise) |
| `DS4_RING_WORTH=2` + `DS4_LA_AHEAD2=1` | 6.07 | -4.0% |
| `DS4_RING_ONLY=4` | 5.41 | -14.6% |

Single runs: `DS4_STORE_EARLY=4` 5.25 vs 5.92 (lost: the early reads crowded the store, "not in the store" rose 80K -> 89K);
`DS4_STORE_PROTECT=1` changed nothing measurable (0 reads dropped, 0 forced). Per distance, a 4-ahead copy paid 46-62%, a next-layer
copy 22-27%, yet copying only 4-ahead lost 15%: a missed next-layer expert lands on the CPU lane on the critical path, which a paid
share per copy does not see. `DS4_RING_WORTH=2` raised the paid copies by a third without raising the speed: the ring is not where the
token's time goes. All stay opt-in; none is re-run as it is. Next: a per-token time breakdown (card, CPU lane, drive waits, attention).
