# Strata-HotRod: the full notes

The README keeps the short version; this is the long one, with every measurement as it was recorded.

## What this fork changes

Upstream Strata, plus (the details and the measurements are in [SPEC.md](SPEC.md)):

- **Stand-in experts.** An expert that is not on the card can be replaced by a close one that is, so the card does not wait
  for the SSD; how far each rank may be swapped is in [the next section](#stand-in-experts-confidence-not-rank). Swap rounds move
  the experts a chat asks for onto the card in the background.
- **Guessing ahead.** A small draft head guesses the next words and the big model checks them together, with
  probabilistic acceptance of the guesses (`--no-spec-prob-accept` turns it off).
- **Three tiers of experts: card -> RAM -> SSD.** Every expert lives in exactly one place. The hottest are on the card, the
  next ones in a RAM tier sized from this PC's free memory, the rest on the SSD; swaps exchange experts between the card
  and RAM, and the look-ahead promotes the experts it predicts from the SSD into RAM.
- **Reading long prompts layer by layer.** The whole prompt passes one layer at a time, so each layer's experts are read
  from the SSD once, not once per piece of the prompt; the reads bypass the page cache (O_DIRECT), and the next layer's
  predicted experts are read ahead while the current one computes.
- **262K context with the K/V in RAM.** From 64K of context the attention's K/V streams from RAM; 32K tokens of it stay
  on the card.
- **Look-ahead on the card.** The next layers' experts are predicted on the graphics card (not recomputed by the CPU)
  while the current layer runs, and the predictions start their reads early. Since 24 Oct the reads ahead use **2 I/O
  threads, the next layer's first** (`STRATA_IO_PF_THREADS`, `STRATA_IO_EDF=0` for the old way): only about 3,000 of
  25,000 experts read ahead per answer were used, and 8 threads' wasted reads queued in front of the decode's own.
  +8% code, +15% reasoning, +8% prose, drive wait 3.8-4.7 s down to 2.8-3.0 s (two runs each, all answers right).
- **STRATA_IO_PF_THREADS=auto (opt-in, untested).** Eight read-ahead workers start, and the guesses in flight (from 2, 1-8)
  follow the decode's own file reads: every 64 reads, above 1.5 times the lowest us-per-MB seen one guess less, below 1.2
  times with a guess held back one more. Demand reads are never capped. Not measured on this box yet.
- **Starting the answer warm.** The experts the end of the prompt used most are put on the card before the first word
  of the answer, so decode does not start cold.
- **Picking up a conversation.** A long conversation's state is saved to the SSD, and its next turn reloads it instead of
  reading the whole conversation again: a 19.5K-token conversation comes back in 0.34 s after another one ran in between.
  The RAM tier first gives back memory for it (in whole huge pages, which Linux frees at once); before that fix 3 of 4
  returns to a 45K conversation were refused for ~100 MiB and read again from the start (68 s). The first answer after
  such a return warms the card with the experts this conversation used (the counts are saved next to it); a conversation
  that stays loaded is left to the swap rounds, which already fit the card to it. The saved conversations are kept to an
  eighth of the free disk; ones an older engine build wrote count against that and go first.
- **Memory safety.** The engine keeps a floor of RAM free before each request; when something else needs RAM, the RAM
  tier gives back its least-used experts until there is room again. The amount it keeps in reserve is measured on this
  PC and learned across restarts (`data/ram-reserve.json`), not a fixed number; the video-memory reserve is learned the
  same way (`vram-reserve.json`).
- **The app:** live speeds while it works, the experts-on-the-card map in the Monitor, and below it the decode timeline
  (Record; double-click or wheel to zoom, drag to pan): each layer's hand-offs and every SSD read, host times only.
- **Experimental, off by default:** `STRATA_GDN_LOOP=1 STRATA_GDN_LOOP_NORM=1` runs the middle linear-attention layers
  twice per token (a training-free "looped" model). Measured on 210 MMLU-Pro questions: 180 correct against 175 without
  it (not significant: only 9 questions differ), and decode 10% slower (38.5 against 42.7 tokens/s). Not worth it on a
  model that was not trained to loop, so it stays off.

## Stand-in experts: confidence, not rank

Default since 24 Oct 2026. Nothing in it is set by hand: the budget, the close-call line and when to re-check are measured.

- **One rule for every rank.** A missing pick takes the best stand-in on the card only when
  `router weight x logit gap <= budget`. A heavy pick (usually rank 0 or 1) needs a stand-in within about a logit; a light one
  (ranks 3-9) can take one much further away; a rank-0 pick in a flat routing is treated as lightly as its weight says.
- **Fragile tokens are decided exactly** (`STRATA_FRAGILE_RECHECK`, opt-in since 9 Nov: `=1` on; off by default after an A/B on Qwen IQ3_S: 42-44 tok/s off vs 31-34 on, answers clean in both). For each token the engine adds up
  how far stand-ins moved it (weight x gap of every replaced pick, all layers). When the token's lead over the runner-up is
  below `k` times that move, the window is cut just before it and the next window decides it with the top ranks exact.
  `k` is learned: the 10th percentile of lead/move on the first 64 windows, then just above the largest move that actually
  flipped a token. One window in 16 also re-checks a token just above `k` (an explore cut), so a flip above the line is seen
  and `k` rises; `k` never falls without a flip seen in this run. On the 24 Oct tests it settled at 0.03-0.05; about 1
  re-check in 5 had flipped (the "bobed", "howed" kind of broken word), and only 1-3% of windows were cut.
- **The budget searches itself** (`STRATA_ROUTE_BUDGET_AUTO`, default on, `=0` off). Windows alternate in runs of 8 between
  budget `b` and `b x step`, so both see the same text; the faster one by 2% over 96 windows each wins, re-check costs
  included, **unless its flip rate is really higher**: with 20+ re-checks on each side, a candidate whose 90% lower bound
  on flips lies above the base's 90% upper bound loses however fast it is. Speed alone once took the budget to 0.95, and
  the text fell apart; the guard stops that climb where stand-ins start changing words.
- **Each request brakes itself** (`STRATA_FRAGILE_BRAKE`, default on, `=0` off). A request starts at half the learned budget.
  When at least 15 of its re-checks show a flip rate above 20% at 90% confidence, its budget halves; after 64 clean windows
  a long answer goes to the full learned budget. Short chats, where stand-ins move tokens about twice as much, stay low.
- **It remembers.** `data/standin-tune.txt` keeps the budget, `k` and the step, tied to the model's fingerprint, and is
  loaded at the next start (`stand-in tuning loaded: ...` in the log).

Measured 24 Oct (131K context; code / reasoning / prose tokens/s, every answer read):

| Setting | Code | Reasoning | Prose | Answers |
| --- | --- | --- | --- | --- |
| the rank rule below (until 23 Oct) | 31.8 | 39.8 | 30.9 | clean |
| fixed budget 0.25, no re-check | 45-51 | 55-57 | 40-41 | broken words in 2 of 3 prose answers |
| self-tuning + re-check, speed-only search, budget 0.29 | 42.8 | 57.0 | 36.8 | clean; at 0.95 later: broken prose, a looping thought in a short chat |
| self-tuning + re-check + guard + brake, fresh start, 3rd round | 37.3 | 48.0 | 29.1 | 27 of 27 clean, a 6-turn short chat included |

Prompt reading keeps the rank rule below (the re-check is decode only).

**Ranks 3-9 take the budget from the first token** (since 24 Oct evening, after malformed JSON and code were seen when
they took any stand-in until the request's first brake). Measured the same evening on one build: 44.7 against 46.4 tokens/s
averaged over 19 answers (about 4% slower, 10% on the 8.5K run), 12 of 12 JSON and code answers valid either way. Two other
rules stay for testing: `STRATA_ROUTE_TAIL_OPEN=1` (ranks 3-9 take any stand-in until the request brakes) and
`STRATA_ROUTE_BRAKE_STEP=1` (ranks 3-9 open, each brake moves the checked ranks up by one; in the 3-question test it gave
a wrong maths answer, 11:24 / 144 km instead of 11:36 / 156 km).

## The look-ahead learns (default since 24 Oct)

The look-ahead guesses, a few layers early, which experts the next layers will route to, so they can be read from the SSD
before they are needed. Until now the guess was the next layer's own router applied to this layer's input. It now
**learns while it runs** (`STRATA_LA_LEARN`, default on, `=0` off):

- **Three heads**, guessing 1, 2 and 4 layers ahead (the farther the guess, the more time the read has).
- **Each guess = the router's own scores + a small learned correction** (a few "directions" per layer). With the correction
  at zero it is exactly the old guess, so it cannot start worse.
- **Trained on the card, every window, from the real routing** (the router's top-32 scores as the target, a hinge keeping
  the rest below), in one batch at the window's end on a side stream.
- **The size chooses itself.** Per layer and head, versions with 4, 8 and 16 directions compete live with the plain router
  on the picks that matter (ranks below the stand-in floor that are not on the card); one replaces the leader only when it is
  better at 90% confidence, then the losers stop training. A leader that falls behind the plain router is reset to it.
- **It remembers**: `data/predictor-<model fingerprint>.bin`, loaded at the next start. Never in git.
- **Only what will be computed is read** (`STRATA_LA_TAIL_READ=2`, default; `1` = ranks 3-9 only, `0` = off): every guess,
  any rank, goes through the router's own stand-in test on the predicted scores, and a pick a stand-in will serve is not read.
  Ranks below the floor are queued first; ranks 3-9 only while the read queue holds at most one guess per I/O worker.

Live on the abliterated IQ3_S (24 Oct, share of the needed experts not on the card that the guesses held):

| Lead | Plain router | Learned |
| --- | --- | --- |
| 1 layer | 77-79% | 85-88% |
| 2 layers | 66-69% | 76-79% |
| 4 layers | 55% | 66% |

3-question test (code / reasoning / prose, sampled, every answer correct), alternated twice on one build: old defaults
59.0 and 54.4 tokens/s on the mean, the learned look-ahead 58.6 and 58.6 (about +3% and steadier); SSD wait 7.0-7.2 against
6.2-6.5 s. Offline replays and the design notes: `docs/PREDICTOR-EXPERIMENTS.md`, `tools/predictor/`.

Opt-in, measured 24 Oct and left off (same 3-question test, every answer correct):

- **Read-ahead gate** (`STRATA_IO_GATE=1`): each guess position (1, 2 or 4 layers ahead, rank 0-9) learns how often its reads
  are used, and a guess is read only when that rate is at least the drive's load ((reads in progress + queued) / workers).
  It won 3% on 19 Oct, before the learned look-ahead. Now 58.5 and 57.4 against 57.7 tokens/s, and the SSD wait rose (8.8 and
  5.9 against 5.4 s): the learned guesses and the stand-in test leave little waste, and what the gate holds back is often needed.
- **Two kernel switches from 19 Oct** (`STRATA_ROUTER_FUSED=1`: the router's GEMV and top-10 in one launch;
  `STRATA_GR_DOWN_WPC=8|4|2`: warps per staged down block) with their byte-exact bench `bench/micro/hotrod_kernel_ab.sh`:
  ported and measured 24 Oct: `bench/micro/hotrod_kernel_ab.sh` byte-identical on 3 seeds (and the split top-k); live,
  3 questions: default 59.3 and 57.7, fused router 58.8, WPC=4 57.5, WPC=2 59.4 tokens/s - within the spread, no gain.

## Stand-in experts, rank by rank (the rule until 23 Oct; prompt reading still uses it)

For every token, each layer's router picks 10 of 512 experts and ranks them, rank 0 the most important. The ones on the
card run at full speed; one that is not there has to come from RAM or the SSD, and that wait is what limits decode. A
**stand-in** is an expert already on the card that the router scored close to the missing one; using it skips the wait.
How close is measured in **logits**, the router's score: a stand-in "within 1 logit" scores at most 1 below the expert it
replaces.

What the install writes for IQ3_S (21 Oct 2026):

| Rank | Not on the card | Why |
| --- | --- | --- |
| **0, 1** (the two most important) | always the real expert, computed by the CPU from RAM, or read from the SSD first | swapping these broke answers (below) |
| **2** | a stand-in only when one scores **within 1 logit**, else the real expert | the closest margin that was safe for rank 1 in the maths test |
| **3 - 9** | the best stand-in on the card, at any distance | these carry little weight; the 50-question maths and 30-question coding test held up |

The settings, in the config's `env`: `STRATA_ROUTE_RESIDENT=1e30` (stand-ins on, any distance),
`STRATA_ROUTE_RESIDENT_RANKS=3-9` (the ranks that take any stand-in) and `STRATA_ROUTE_RESIDENT_EDGE=1` (the rank just above
them, rank 2, only within 1 logit). Measured with this rule (21 Oct, 131K context): 53.6 tokens/s on the 8.5K-prompt run,
the same as with ranks 2-9 taking any stand-in.

The same rule also works in two more places (both on by default since 21 Oct):

- **The look-ahead skips what a stand-in will cover.** The look-ahead reads ahead the experts the next layers will want.
  A predicted rank-2 expert with a stand-in already on the card within 1 logit is no longer read: the router will take
  the stand-in. On the 8.5K run that skipped about 47,000 reads, cut the experts read ahead from 16,968 to 12,009 and the
  decode's wait for the SSD from 4.7 to 4.0 s: 56.4 against 53.6 tokens/s, and 9 to 13% faster on short prompts.
  **Off by default since 24 Oct** (`STRATA_LA_EDGE_SKIP=1` turns it on): it assumes the rank-2 stand-in rule, but decode now
  follows the confidence budget, which often computes those picks exactly, so the skipped reads became misses (reasoning
  waited 11.8 against 5.2 s; with it off 30.2 / 42.9 / 30.6 against 28.4 / 32.3 / 23.1 tokens/s). It only changes what is
  read ahead, never the answer.
- **Reading the prompt uses stand-ins too.** The prompt is read with the same rule, so a short prompt reads far fewer
  experts from the SSD: 76 to 118-token prompts are read in 1.3 to 1.8 s instead of 2.3 to 2.8 s (35 to 45% faster); an
  8.5K prompt 15.1 against 15.8 s. This changes what the model computes from the prompt; the answers checked (code, a
  maths problem, prose, an 8.5K summary) read clean. `STRATA_PREFILL_STANDIN=0` turns it off.

What we found on the way (20 Oct 2026, every answer read by hand):

| Swapping the top two ranks | Result |
| --- | --- |
| at any distance | answers collapse: empty replies and loops |
| within 3, 5 or 8 logits | collapse: "Let's solve step-by step." repeated, "salt, salt, salt ..." |
| within 2 logits | dropped tokens (a missing `]` in code), broken words, and wrong sums in maths |
| within 1 logit, rank 1 only | 49 of 50 maths answers right, the same as with nothing swapped |
| within 1 logit, ranks 0 and 1 | faster (47.4 against 38.6 tokens/s on short prompts), but leaked a turn marker into an answer at 94K tokens |
| rank 1 within 1 logit, on a card just after a restart | broken answers in an agent session, so the top two are now always real |

The opt-in switches for the top two ranks stay in the engine for testing: `STRATA_ROUTE_RAM_FLOOR=1` (rank 1 within
`STRATA_ROUTE_RAM_FLOOR_MARGIN` logits, default 1) and `STRATA_ROUTE_RAM_FLOOR_RANK0=1` (rank 0 too).

## What the per-layer learning settled on (Qwen3.8-Flash-Next IQ3_S, 23 Nov 2026)

One machine (RTX 3060 12 GB, 31 GB RAM, PCIe Gen3 x4 NVMe), 300-token requests, the default stand-ins (2E1: floor 3, edge 1).

**The weight thresholds** (the system-driven default, skip share 0.5): per layer between **0.078 and 0.169**, which is x3.9 to x8.5 of the flat 0.02, settled at window 384 with the largest move 0.3-0.5%, and the same to within 0.001-0.005 across three restarts (0.0840..0.1506, 0.0839..0.1542, 0.0834..0.1554 in the early runs).  Once settled nothing is rewritten while the thresholds stay within 3%.

**Full learning** (`STRATA_LAYER_LEARN=full`): a layer that has not settled runs exact and measures what dropping a missed pick would cost.  On a fresh state 38 of the 48 layers settled in 90 seconds, 45 in about 4 minutes (the rest skip rarely and are cheap to leave exact); the layers that settled at the 64-pick minimum had a standard error of 6-10%.  Restarted with the saved state, 46 layers started settled and none had to learn again; none woke.  The measured cost of dropping a pick (router-weighted output size) **grows with depth**:

| layers | weighted skip error |
|---|---|
| 0-14 | 0.08-0.18 |
| 15-30 | 0.11-0.26 (most 0.12-0.18) |
| 31-35 | 0.21-0.32 |
| 36-47 | 0.46-0.67 (highest layer 42, 0.67) |

so a skip in the last twelve layers costs 3-5 times what it does in the middle ones.  `STRATA_LAYER_SUB=1` uses this to give the expensive layers a tighter stand-in budget.

**Coherence and accuracy, as measured.**  The 13 checkable problems (arithmetic word problems, the answer on a final `ANSWER:` line) were answered **13 of 13 correctly** on the plain defaults, with the stand-in budget scaled per layer, and with full learning; I read the replies (the short ones whole, the long ones at their start and end) and found no loops, stray or leaked tokens, broken words or cut-off text, and re-did several by hand (167 x 699 + 363 mod 8 = 0; 12 pens at $3 and 9 notebooks at $16 from $200 leave $20).  The three runs differ in wording and length, as a sparse model does.

**What this does not show.** It is a small, easy set: it cannot see a token that flips and then recovers, and no run compared the output token by token with an exact (no-skip) run, so "no flips" is **not** shown; the safe skip share is therefore still the author's start value (0.5), not a measured limit.  One request made by hand on the full-learning build ended after 40 tokens with the model's own stop token; the text was not seen, so whether that was a short answer or damage is not known.  A settle at 64 picks may be optimistic for a heavy-tailed cost.  The saved state has no model fingerprint yet (a different model with the same layer count would inherit it), so learning is opt-in.  The low-rank copies (`STRATA_LOWRANK=1`) cut the error of layer 0 to 0.885 of skipping's in the engine against 0.55 offline, for copies covering 44% of that layer's skipped picks: unexplained, and not recommended beyond testing.

## Switches (moved from the README, 24 Oct)

| On by default | Off: `=0` |
| --- | --- |
| `STRATA_ROUTE_BUDGET_AUTO`, `STRATA_FRAGILE_RECHECK`, `STRATA_FRAGILE_BRAKE` | the self-tuning stand-ins |
| `STRATA_LA_LEARN`, `STRATA_LA_TAIL_READ=2` | the learned look-ahead, need-only read-ahead |
| `STRATA_PIN_SEGS` | the pageable RAM tier page-locked in 1 GiB segments (no extra RAM; a give-back unlocks a segment first). With `STRATA_PCIE_BALANCE`: 39.7 vs 36.5 tok/s (+9%), 38.2 pinned alone, no copy failures (RTX 3060 PC) |
| `STRATA_PCIE_BALANCE` | each verify layer's PCIe count from measured CPU and GPU costs (one GPU, native pack, no batch slots) |
| `STRATA_PCIE_EARLY` (on by default on CUDA; `=0` off; on AMD/HIP still opt-in `=1`, not measured there; a latent pack (the Edge0 model) takes its own layer path and does not use it yet) | the PCIe share's fetch kernel runs on its own stream beside the layer's VRAM experts instead of after them (PCIe mode 2, the default copy-kernel path). Measured on an RTX 3060 PC, Qwen UD-Q4_K_XL, stage timers on: `pcie-experts` 19.9 -> 1.8 ms per window, `vram-experts` 10.2 -> 27.7 ms (the two now overlap), `cpu-wait` 34.6 -> 28.6 ms; about 7% off the window, 1-4% more tok/s, 99th-percentile token gap about 13% lower |
| `STRATA_PCIE_STATS=1` (opt-in, diagnostics only) | every 20000 layer calls: distinct misses, how many the PCIe share asked for and how many could fly (only a page-locked RAM-tier blob can), the refusals split into RAM-tier-not-pinned and file-tier (93% of the refusals in one run were file-tier blobs), how repeated the file-tier misses are (the share the N most-missed experts cover and the net of swapping them for the N least-missed RAM-tier ones), the CPU lane's `jobs` phase bucketed by the layer's file-tier misses (about 0.9 ms per file-tier miss: the drive's bandwidth), and the promotion funnel when `STRATA_PROMO_NETGAIN` is on. Costs a mutex and a table update per miss: do not leave it on for speed runs |
| `STRATA_PROMO_NETGAIN=1` (opt-in, measured: no speed gain) | the RAM tier's look-ahead promotion by decayed per-expert miss score (>= 1, half-life about 140 windows) against the lowest-scoring RAM-held expert of the layer, instead of the victim's usage < 1.0; a promotion unused for 300 windows may be a victim again. 12x the promotions (about 1,700-2,100 per request, 60-75% used), file-tier demand misses down 10-14%, but tok/s unchanged (the drive is bandwidth-bound: about 0.9 ms per file-tier miss, and the total bytes read stayed the same) |
| `STRATA_PROMO_GUESS_MIN=<score>` (opt-in, needs `STRATA_PROMO_NETGAIN=1`; measured: slightly slower) | a decode window's read-ahead guess for an expert whose miss score is below `<score>` is not read. With `1`: the OS read per pass fell 35% (156-186 GB to 101-125 GB) and the file-tier demand misses 26%, but tok/s was 21.1 against 21.6 over six mirrored legs each (p99 gap and stalls slightly worse): the skipped guesses were partly the useful ones, whose misses are then exposed demand reads. Kept for slower or shared drives, where the traffic saving may matter |
| `STRATA_TAIL_CPU=0` (the rank rule: the default of 0.1.45 to `ratrod.v1`, now replaced by the weight rule below unless `STRATA_TAIL_WEIGHT=0`; `STRATA_TAIL_CPU=1` turns both tail rules off; +25% in one early run with the mask off, no gain in later paired runs) | the rank rule: the ranks from the stand-ins' floor up are never computed by the CPU or read over PCIe: a missed expert that every token of the window routes at or above the floor is skipped and its weight dropped (no renormalisation). Needs the stand-ins (`STRATA_ROUTE_RESIDENT`); answers differ slightly |
| `STRATA_TAIL_READ=0` (opt-in, with `STRATA_TAIL_CPU=0`; measured: no difference) | the read-ahead leaves the guesses at the floor rank and above unread. The router already drops the guesses a resident stand-in serves, so little is left to skip |
| `STRATA_RANK_CUTOFF=R` (opt-in, 1-9; `STRATA_RANK_CUTOFF_CLIP=0` keeps the stand-ins at ranks R-9; measured slower in the same session) | ranks R-9 are not computed at all, resident or not, and the read-ahead does not read them; the stand-in range ends at rank R-1 |
| `STRATA_TAIL_WEIGHT=<tau>` (default 0.02 when the stand-ins exist and some experts are not resident; edit it: 0.05, 0.10 or any 0 < tau < 1; `0` turns the weight rule off and the rank rule stays; with `STRATA_TAIL_CPU` unset it replaces the rank rule, with `STRATA_TAIL_CPU=1` and `STRATA_ROUTE_TAIL_SKIP=0` it stands alone) | a missed expert (no stand-in took its place) that every token of the window routes at or below the stand-ins' floor rank (taken from `STRATA_ROUTE_RESIDENT_RANKS`: rank 3 in the shipped config, so ranks 0-2, the edge rank 2 included, are always computed; rank 2 when there are no stand-ins) with a router weight below `<tau>` is neither copied nor computed and its weight is dropped (no renormalisation). Chosen by weight, not rank: offline on this model (4 layers, 1,600 decode rows each, exact expert outputs) dropping the smallest weights costs about half the block-output error of dropping the highest ranks at 10% of picks dropped (0.047 against 0.098), and the router weight is within about 0.01 of the true contribution as a ranking. CORRECTION (24 Nov): until the fix of that day the default (single-GPU) path never copied the routing weights to the host, so this rule saw a weight of 0 for every pick and skipped every missed tail expert at or below the floor whatever its weight and whatever tau was set to - it behaved as the rank rule. The 23 Nov measurements of it (48 problems: 28 correct against 26 exact; 22 tok/s; the same speed at 0.02, 0.05 and 0.10; the terse replies; the skipped counts) are measurements of that rank behaviour, not of a weight rule. With the weights published, tau 0.02 skips far fewer picks (offline on the real weights about 0.24 picks a token over all layers), so the default decodes slower than the earlier numbers until a higher tau or `STRATA_TAIL_CPU=0` (the rank rule) is chosen; it has not been re-measured. Answers differ |
| system-driven weight rule (the default with the default stand-ins and no explicit `STRATA_TAIL_WEIGHT`; `STRATA_LAYER_LIVE=0` turns it off; `STRATA_LAYER_SHARE=<0..1>`, default 0.5) | the engine skips the lowest-weight `SHARE` of the missed tail picks (ranks at or above the stand-ins' floor that no stand-in took) and spreads them over the layers: it keeps a running log-binned histogram of the missed picks' router weights per layer (saved beside the expert profile in `layer-live.bin` every ~1000 windows and at exit, loaded at start), takes each layer's shape from the mean weight of its weakest 10% (a layer whose weakest picks carry more weight loses more from skipping them; offline on the real weights this had 0.52 of the flat rule's block-output error proxy at the same skip count, 0.44 with the exact map), and re-solves the thresholds every 64 windows so the total skipped is `SHARE`. The `0.02` setting plays no part; with `STRATA_LAYER_LIVE=0` it is a plain threshold on every layer (and then skips almost nothing: the weights of rank 3-9 picks sit between 0.04 and 0.12, tau 0.02 is below 0.1% of them, 0.05 is 11%, 0.08 is 64%, 0.10 is 90%). Until about 2,000 weights have been seen (seconds) it skips nothing. The share 0.5 is a choice, not a measured optimum |
| `STRATA_LAYER_SUB=1` (opt-in) | the stand-in budgets (`w x gap <= budget`) scale per layer with the system-driven shape (the multipliers divided by their mean, so the overall budget stays the user's): a layer where skipping costs little takes looser stand-ins, a sensitive one refuses more. Measured (23 Nov 2026, 10 requests of 300 tokens each, one machine): 5,013 skipped experts per request on the plain defaults against 3,771 with it (25% fewer skips), speed within the spread between requests (18.3 against 18.6 tok/s; the plain run was measured while an offline job shared the CPU, so the speed is not a verdict), 13 of 13 checkable problems right on both.  Offline, no free signal picks a better stand-in than a random resident expert and a stand-in never beat skipping at the block output, so this only moves how many stand-ins are taken per layer |
| `STRATA_LAYER_LEARN=full` (opt-in; `=1` is the same; `=sample` the other strategy; unset/`0` off) | per-layer learning of what skipping costs.  **full** (the default mode): a layer that has not settled runs exact (the weight rule skips nothing there; the stand-ins run as always) and every missed pick the CPU computes anyway is a free measurement - its router weight and output size, `w^2 |output|^2`, and where the layer has low-rank copies the copy's error beside it.  When the layer's estimate is stable (standard error of the mean within 10%, at least 64 picks, capped at 4096) it **settles**: skipping turns on there with its own threshold and nothing is measured any more; layers settle one by one (on Qwen 38 of 48 in 90 seconds, 45 in 4 minutes).  A settled layer wakes only if its live threshold moves more than 10%.  The state is saved in `<data dir>/layer-shadow.bin` and settled layers start settled after a restart.  **sample**: skipping runs from the start and 1 skipped expert in a few windows is also computed exactly and compared (every 6th window until the layer's first 64 samples - once, ever - then every 16th); the same settle rule.  Measured on Qwen: the skip cost grows with depth (weighted skip error 0.08-0.18 in layers 0-14, 0.2-0.3 around 31-35, 0.45-0.67 in layers 36-47).  Not yet: a model fingerprint in the saved state (a different model with the same layer count would inherit it), so it stays opt-in.  Your own values win: `STRATA_LAYER_LIVE=0` with `STRATA_TAIL_WEIGHT` and the 2E1 setting give the fixed rule with nothing learned. |
| `STRATA_LOWRANK=1` (opt-in, with `STRATA_LOWRANK_LAYERS=<list>`) | a skipped pick that has a data-aware low-rank copy in `<data dir>/lowrank.bin` (`tools/lowrank_fit.py`, fitted in idle time by `serve/lowrank_maint.py`, which settles when the copies cover 90% of the skipped picks and wakes if the traffic shifts) is computed from the copy on the CPU instead of being zero; the per-expert skip counts that choose the experts to fit run only with this switch.  Measured: offline the copy's error is 0.55 (layer 0, rank 128, 4-bit) to 0.8 (layer 24) of skipping's; in the engine layer 0 showed 0.885 of skipping's with copies for 44% of the skipped picks - not yet explained, so unproven. |
| `STRATA_LAYER_SUB=1` with learning on | the stand-in budget multiplier per settled layer comes from its measured skip cost (inverse of the cost, mean 1 over the settled layers, x0.25..x2) instead of the weight shape: a layer where dropping a pick is expensive gets a tighter budget.  Pushed to the kernel again only when a layer settles or wakes. |
| request keys `strata_tune: {"tail_weight": <tau>, "tail_skip": <R>}` (the audit's switches, `tools/quality_audit.py`) | set `STRATA_TAIL_WEIGHT` and the tail skip rank for one request only (0 = off); the next request without them gets back the value that was in force. Nothing changes while no request uses them |
| `STRATA_LAYER_SENS=<file>`, else `<pack>/layer-sens.txt` beside the expert profile when it exists (watched: a file that appears or changes is loaded at the next request, no restart; one number per layer, larger = the layer suffers more from a dropped expert; `#` lines ignored). `tools/layer_table.py` makes and refreshes it from a `STRATA_ROUTER_TRACE` capture of your own traffic and the model's GGUF: each run blends its measurement into the table already there (exponential average) and `--part k/n` measures every n-th layer, so a short nice'd job can run now and then and the table improves with use. No file: every layer keeps the flat threshold, the shipped default | scales the weight rule's threshold per layer, `tau_l = tau * m_l`, `m_l` proportional to 1 / sensitivity (clamped 0.25 to 4, mean 1 so the average threshold stays `tau`): a tolerant layer skips more of its small picks, a sensitive one fewer. Pseudo-RCO: RCO (arXiv 2605.00649, used with the GSQ quantizer, arXiv 2604.18556) spends a fixed budget unevenly across layers and experts, searched against the model's own calibration loss; this spends the skip budget unevenly too, from a measured per-layer error map instead of a search. A first file for the Qwen model (block-output error with 10% of the picks dropped by weight, 48 layers, 1,600 decode rows) gives `tau = 0.02` the range 0.011 to 0.069: layers 0 and 47 tolerate the most, the middle layers the least. Not yet measured end to end (the audit that would show a difference is still too noisy); off unless the file is given |

| Opt-in, measured and left off | Why |
| --- | --- |
| `STRATA_ROUTE_TAIL_OPEN=1` | ranks 3-9 unchecked until a brake: faster, malformed JSON/code seen |
| `STRATA_ROUTE_BRAKE_STEP=1` | each brake checks one more rank: a wrong maths answer |
| `STRATA_IO_GATE=1` | read-ahead gate by drive load: no gain with the learned look-ahead |
| `STRATA_LA_EDGE_SKIP=1`, `STRATA_LA_MODEL=<file>` | superseded by the learned look-ahead |
| `STRATA_GDN_LOOP=1` | looped middle layers: 10% slower, no clear accuracy gain |
| `STRATA_ROUTER_FUSED=1`, `STRATA_GR_DOWN_WPC=4\|2` | byte-identical outputs; alone and together within run-to-run spread: no gain |

## Earlier speed tables (moved from the README)

Writing answers, 17 Oct 2026 (three prompts, each asked three times; the repeats are faster because the prompt's experts
are already on the card):

| Prompt (answer length) | 128K context, first ask | 128K, repeats | 32K context, first ask (5 runs) |
| --- | ---: | ---: | ---: |
| Code (240 - 320 tokens) | 43.9 tokens/s | 67.9, 76.1 | 43.0 - 44.0 |
| Reasoning (790 - 900 tokens) | 53.7 tokens/s | 64.1, 71.7 | 53.7 - 61.8 |
| Story (140 - 180 tokens) | 41.8 tokens/s | 43.1, 51.1 | 36.1 - 39.8 |

Short prompts, seconds until the first word (10 Oct 2026, before the 0.1.45 merge): 150-260 tokens 3.8-5.8 s, 430-500 tokens
7.5-10.5 s, 1,600-2,100 tokens 14-16 s.

- **STRATA_IO_PF_THREADS=auto, measured 6 Nov (lost, stays opt-in).** 3 alternated rounds at 131K, the three calibration questions:
  default (2 threads) 34.7 tok/s, auto 32.7 (-5.6%). The cap climbed to 5-7 on the long reasoning answer, where it lost 4-17%: the
  decode's reads slowed only ~22% per MB (under the 1.5x step-down) while the decode lost ~10%, so read time per MB does not see
  what extra guesses cost on Qwen. A variant worth building: step the cap by answer speed itself (as DeepSeek's DS4_RING_INFLIGHT=auto).

- **STRATA_MASK=<share> (opt-in, changes the output).** Per layer, the least-used experts of the expert profile (the pairs it ranks
  last, or does not rank) are masked at the router. After the top-10, a masked pick of rank >= STRATA_ROUTE_RESIDENT_RANKS lo moves
  to the best unmasked expert the token did not pick; rank lo-1 moves only within STRATA_ROUTE_RESIDENT_EDGE; ranks below lo-1 stay.
  The look-ahead's guesses follow the target layer's mask. Each move adds its weight times the logit gap to the row's row_shift, so
  STRATA_FRAGILE_RECHECK re-decides the close calls the mask makes; the re-check's exact re-run runs unmasked. Unset: nothing runs.
  Request line: `strata serve: mask (STRATA_MASK): <seen> masked picks seen, <moved> moved`. Built, not run.

- **STRATA_MASK=learn (opt-in, untested).** The mask learns from the model's own divergence on the user's traffic (the DeepSeek port of
  DS4_MASK=learn). It starts at a 5% share (the profile's least-used pairs). A flip re-check cuts a decode window at row r; the next
  window runs exact, and the KL between its row 0 and the cut row's first pass (D) is shared over the row's replaced picks as harm; every
  replaced pick of a decode row is exposure. Harm rate = harm / exposure, decayed by half every 8192 tokens. Every 256 decode tokens the
  share follows the re-check agreement (98% or more: up 2.5 points; under 95%: down 5 points) and the masks move among the least-used
  pairs by rate; a pair with 256 or more exposures and a rate under its layer's 10th percentile is masked at every rank. The table is
  saved as `mask-learn-<model>.bin` beside the stand-in tuning file, every 4096 tokens and at each request's end. Needs the flip re-check
  and route-resident. The look-ahead learner's loss gives an expert masked at every rank no weight (its gradient is zero), so the learning
  spends itself on the experts that can be needed. Request line: `strata serve: mask learn: p <p>, g <g>, <m> masked (<a> at all ranks), re-check D mean <d> (<n>), <k> experts scored`.
  The depth is two numbers: the share p and g, the fraction of the masked experts allowed at level 2 (all ranks): at most round(g x the
  layer's masked count) of a layer's graduated pairs, the lowest harm rates first. g starts at the loaded table's level-2 share (else 0.25).
  With STRATA_ROUTE_BUDGET_AUTO on, the budget tests alternate with mask tests (`STRATA_MASK=learn` x AUTO): the same interleaved arms, both
  at the request's budget, base (p, g) against one step deeper (p + 0.025, g + 0.1) or shallower (minus), starting shallower. A candidate
  wins at 1.02 times the base's speed with the same quality rule as the budget test; a win keeps its (p, g). While a mask test runs the share
  and the levels do not move (the harm and exposure keep learning). The arm's levels are uploaded when the arm changes. Request line:
  `strata serve: auto mask test (p,g) cand vs (p,g) base: <x> vs <y> tok/s over <a>+<b> windows -> (p,g); flip rate ...`.

- **STRATA_STANDIN_LEARN=1 (opt-in, changes the output).** Each stand-in learns a correction for its (layer, true expert, stand-in)
  pair from the flip re-checks. A cut window keeps the row's first stand-in output y_s (a device copy per layer, gated by the host
  between windows); the exact successor's row 0 gives the true expert's y_e at the same layer and rank. Per pair, a decayed least
  squares over the channels (c = num / den, clipped to [0.5, 2], used after 32 samples) scales the stand-in's routing weight in the
  route-resident kernel. Pairs (LRU 4096 per layer) persist in standin-learn-<fp>.bin beside standin-tune.txt. Needs
  STRATA_FRAGILE_RECHECK and the whole model on one GPU; otherwise off. Request line: `strata serve: stand-in learn: <n> pairs ready,
  <s> samples this request, mean |c-1| <v>`. Built, not run.

- **STRATA_STANDIN_BALANCE=1 (opt-in, needs STRATA_STANDIN_LEARN, changes the output).** A ready pair's risk in the re-check is its
  weight times the logit gap times clip(err / mean err, 0.25, 4), where err = sqrt((ee - 2c num + c^2 den) / ee) and the mean is over the
  layer's ready pairs. Every 256 committed decode tokens each layer ranks its experts by use x best err / (decayed demand misses + 1).
  The top decile gets an eviction factor of 4 (kept in the RAM tier), the bottom decile 0.25 (given up first); one step a round, and back
  to 1 only after two rounds on the other side. The factors scale FileExpertSource's coldest-resident choice and the refill's cold list.
  Request line: `strata serve: stand-in balance: <r> rounds, <k> experts kept fast, <g> given up`. Built, not run.
